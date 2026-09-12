"""Track B — controlled IB=RG with a from-scratch Variational Information Bottleneck.

We SET the compression beta and watch, with ground-truth causal/nuisance factors,
whether the representation that keeps the causal variable and discards the nuisance
is the one that generalizes.

Canonical IB setup (why: a nuisance must be able to HURT generalization, else
discarding it is invisible — see NEXT_PHASE / the "why shortcuts" discussion):
  x = [causal block | nuisance block].  y depends ONLY on the causal block.
  TRAIN labels carry noise (a fraction are flipped); TEST labels are clean.
  The nuisance block is a high-rate channel the model can use to MEMORIZE the
  flipped train labels. So:
    low beta (little compression)  -> memorizes noise via nuisance -> overfits,
                                      learns the causal signal less well -> worse OOD.
    high beta (much compression)   -> can't afford the nuisance channel -> forced
                                      onto the causal signal -> generalizes.
Predictions: as beta rises, nuisance is discarded (recon of the nuisance from z
falls), causal is preserved, the train-test gap closes, and OOD test accuracy
improves up to an optimum (the IB phase transition). CPU-only; minutes per seed.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score

BETAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def make_data(rng, n, protos, n_u, n_classes, noisy):
    n_c = protos.shape[1]
    y = rng.integers(0, n_classes, n)
    Xc = protos[y] + 0.8 * rng.standard_normal((n, n_c))       # causal (generalizes)
    Xu = rng.standard_normal((n, n_u)).astype(np.float32)      # nuisance / memorization channel
    X = np.concatenate([Xc, Xu], axis=1).astype(np.float32)
    y_obs = y.copy()
    if noisy:                                                  # train: flip a fraction of labels
        flip = rng.random(n) < 0.25
        y_obs[flip] = rng.integers(0, n_classes, int(flip.sum()))
    return X, y_obs.astype(np.int64), y.astype(np.int64)       # (X, observed y, clean y)


class VIB(nn.Module):
    def __init__(self, d_in, z_dim, n_classes, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, 2 * z_dim))
        self.cls = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
        self.z_dim = z_dim

    def forward(self, x, sample=True):
        h = self.enc(x)
        mu, logvar = h[:, :self.z_dim], h[:, self.z_dim:]
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
        return self.cls(z), kl, mu


def _probe_acc(Z, y):
    h = len(Z) // 2
    return float(balanced_accuracy_score(y[h:], LogisticRegression(max_iter=2000).fit(Z[:h], y[:h]).predict(Z[h:])))


def _recon_r2(Z, Y):
    h = len(Z) // 2
    return float(Ridge(alpha=1.0).fit(Z[:h], Y[:h]).score(Z[h:], Y[h:]))


def run_beta(beta, data, n_c, cfg, seed):
    Xtr, ytr, _ = data["train"]; Xte, yte, yte_clean = data["test"]
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
    Ztr, Zte = mu_tr.numpy(), mu_te.numpy()
    train_acc = float((lg_tr.argmax(1).numpy() == ytr).mean())
    test_acc = float((lg_te.argmax(1).numpy() == yte_clean).mean())         # OOD (clean labels)
    return {
        "beta": beta, "rate": float(kl_tr.mean()),
        "train_acc": train_acc, "test_acc": test_acc, "gap": train_acc - test_acc,
        "nuisance_recon": _recon_r2(Ztr, Xtr[:, n_c:]),                       # z keeps nuisance?
        "causal_retention": _probe_acc(Zte, yte_clean),                      # z keeps causal?
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--n_c", type=int, default=4)
    ap.add_argument("--n_u", type=int, default=20)
    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"seed{args.seed}"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    rng = np.random.default_rng(args.seed)
    protos = rng.standard_normal((args.n_classes, args.n_c)) * 2.0           # shared causal structure
    cfg = {"n_classes": args.n_classes, "z_dim": args.z_dim, "epochs": args.epochs}
    data = {"train": make_data(rng, args.n_train, protos, args.n_u, args.n_classes, noisy=True),
            "test": make_data(rng, args.n_test, protos, args.n_u, args.n_classes, noisy=False)}
    sweep = []
    for b in BETAS:
        r = run_beta(b, data, args.n_c, cfg, args.seed)
        sweep.append(r)
        print(f"[vib] beta={b:<6} R={r['rate']:.2f} train={r['train_acc']:.3f} test={r['test_acc']:.3f} "
              f"gap={r['gap']:.3f} nuis_recon={r['nuisance_recon']:.3f} causal_ret={r['causal_retention']:.3f}", flush=True)
    best = max(sweep, key=lambda r: r["test_acc"])
    print(f"[vib] best OOD test={best['test_acc']:.3f} at beta={best['beta']} vs beta0 test={sweep[0]['test_acc']:.3f}", flush=True)
    odir = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(odir, exist_ok=True)
    json.dump({"seed": args.seed, "cfg": vars(args), "sweep": sweep},
              open(os.path.join(odir, f"vib_{tag}.json"), "w"), indent=2)
    print(f"[saved] out/vib_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
