"""Track B — controlled IB=RG with a from-scratch Variational Information Bottleneck.

We SET the compression beta and watch, with ground-truth causal/nuisance/shortcut
factors, whether the representation that preserves the causal variable and discards
the nuisance is the one that generalizes OOD.

Task: x = [causal block | nuisance block | shortcut dim].
  y depends ONLY on the causal block. The shortcut dim = y at TRAIN (a perfect
  spurious cue) but is RANDOM at TEST (do(U) recombination). A model that leans on
  the shortcut gets high train / low test accuracy.

VIB: enc q(z|x) -> z -> classifier p(y|z);  loss = CE + beta*KL(z).  KL = the RATE.

Per beta we measure (all with ground truth):
  R (rate), train_acc, test_acc (OOD),
  shortcut_retention = decodability of the (test, decorrelated) shortcut value from z
                       -> does z still ENCODE the nuisance shortcut, or discard it?
  causal_retention   = decodability of y from z.
IB=RG predictions: as beta rises, shortcut_retention falls (nuisance discarded) while
causal_retention is preserved; the beta that best discards the shortcut is the beta
that generalizes best; the rate-distortion curve may show a phase transition.
CPU-only; a full beta-sweep runs in ~minutes.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

BETAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]


def make_data(rng, n, n_classes, n_c, n_u, reliable):
    y = rng.integers(0, n_classes, n)
    protos = rng.standard_normal((n_classes, n_c)) * 2.0
    Xc = protos[y] + rng.standard_normal((n, n_c))                 # causal: class-conditional
    Xu = rng.standard_normal((n, n_u))                            # nuisance: independent of y
    sc = y.copy() if reliable else rng.integers(0, n_classes, n)  # shortcut value
    Xs = (sc[:, None] + 0.1 * rng.standard_normal((n, 1))).astype(np.float32)  # near-perfect cue
    X = np.concatenate([Xc, Xu, Xs], axis=1).astype(np.float32)
    return X, y.astype(np.int64), sc.astype(np.int64)


class VIB(nn.Module):
    def __init__(self, d_in, z_dim, n_classes, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, 2 * z_dim))
        self.cls = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
        self.z_dim = z_dim

    def forward(self, x, sample=True):
        h = self.enc(x)
        mu, logvar = h[:, :self.z_dim], h[:, self.z_dim:]
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)   # per-sample rate (nats)
        return self.cls(z), kl, mu


def probe(Z, labels):
    n = len(labels); half = n // 2
    clf = LogisticRegression(max_iter=2000).fit(Z[:half], labels[:half])
    return float(balanced_accuracy_score(labels[half:], clf.predict(Z[half:])))


def run_beta(beta, data, cfg, seed):
    Xtr, ytr, _ = data["train"]; Xte, yte, scte = data["test"]
    torch.manual_seed(seed)
    m = VIB(Xtr.shape[1], cfg["z_dim"], cfg["n_classes"])
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(cfg["epochs"]):
        m.train(); opt.zero_grad()
        logits, kl, _ = m(xt)
        (F.cross_entropy(logits, yt) + beta * kl.mean()).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        lg_tr, kl_tr, mu_tr = m(xt, sample=False)
        lg_te, _, mu_te = m(torch.tensor(Xte), sample=False)
    Zte = mu_te.numpy()
    return {
        "beta": beta,
        "rate": float(kl_tr.mean()),
        "train_acc": float((lg_tr.argmax(1).numpy() == ytr).mean()),
        "test_acc": float((lg_te.argmax(1).numpy() == yte).mean()),   # OOD generalization
        "shortcut_retention": probe(Zte, scte),                        # nuisance kept in z?
        "causal_retention": probe(Zte, yte),                           # causal kept in z?
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--n_c", type=int, default=6)
    ap.add_argument("--n_u", type=int, default=6)
    ap.add_argument("--z_dim", type=int, default=8)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"seed{args.seed}"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    rng = np.random.default_rng(args.seed)
    cfg = {"n_classes": args.n_classes, "z_dim": args.z_dim, "epochs": args.epochs}
    data = {"train": make_data(rng, args.n, args.n_classes, args.n_c, args.n_u, reliable=True),
            "test": make_data(rng, args.n, args.n_classes, args.n_c, args.n_u, reliable=False)}
    sweep = []
    for b in BETAS:
        r = run_beta(b, data, cfg, args.seed)
        sweep.append(r)
        print(f"[vib] beta={b:<6} R={r['rate']:.2f} train={r['train_acc']:.3f} "
              f"test={r['test_acc']:.3f} shortcut_ret={r['shortcut_retention']:.3f} "
              f"causal_ret={r['causal_retention']:.3f}", flush=True)
    best = max(sweep, key=lambda r: r["test_acc"])
    print(f"[vib] best test_acc={best['test_acc']:.3f} at beta={best['beta']} "
          f"(shortcut_ret there={best['shortcut_retention']:.3f})", flush=True)
    odir = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(odir, exist_ok=True)
    json.dump({"seed": args.seed, "cfg": vars(args), "sweep": sweep},
              open(os.path.join(odir, f"vib_{tag}.json"), "w"), indent=2)
    print(f"[saved] out/vib_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
