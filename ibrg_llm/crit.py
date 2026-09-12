"""Track B — Foliation & criticality: the empirical capstone (P1).

One beta-sweep on the controlled VIB that produces, per beta, THREE measurements —
the three legs of the thesis "the causal/spurious distinction is invisible to
observation/geometry and visible only to interventional response, and is maximal at
the compression critical point."

Task (same shortcut world as boundary.py):  x = [causal block | shortcut block].
  y depends ONLY on the causal block (learnable-but-noisy).
  The shortcut is a strong one-hot cue.  Two training conditions:
    "aligned"  — shortcut == y on ALL train data (a single observational distribution;
                 shortcut is a perfect, cheap, minimal-sufficient statistic -> IB cannot
                 tell it apart from the causal feature; OOD stays broken at every beta).
    "recomb"   — do(U): shortcut == y on half the train data, random on the other half
                 (the intervention). The model is forced to also learn the causal route.
  OOD/eval environments vary rho = P(shortcut == y) from 1.0 (aligned) to 0.0 (anti),
  so a feature's relationship to y can be tested for cross-environment INVARIANCE.

We build a ground-truth-labelled bank of directions in z: a "causal" direction is the
z-shift induced by moving the CAUSAL input block between two classes (shortcut fixed);
a "spurious" direction is the z-shift from moving the SHORTCUT block (causal fixed).
Per direction v and per beta we compute:

  E1 FOLIATION
    observational score   = Y-decodability of <z,v> on the TRAIN distribution.
                            Prediction: causal ~ spurious  => obs-AUC ~ 0.5 (can't tell).
    invariance score      = min over eval environments of Y-decodability of <z,v>
                            (a causal feature predicts y in EVERY environment; a
                            spurious one only where shortcut~y). This is the
                            cross-environment / interventional signal.
                            Prediction: causal >> spurious  => inv-AUC >> 0.5 (revealed).

  E2 CRITICALITY
    separation(beta)      = mean invariance(causal) - mean invariance(spurious), and
                            inv-AUC(beta). Prediction: peaks at beta* (the generalization
                            optimum / rate-distortion kink located from the OOD curve).

  E3 FLUCTUATION-DISSIPATION (descriptive)
    susceptibility chi    = mean |d logit_correct| under a small do(z += eps*v) push
                            (the linear response of the classifier to a latent kick).
    fluctuation Var(z)    = mean total variance of mu on OOD data (trace of Cov).
                            Prediction: chi and Var(z) track / coincide near beta*.

CPU-only; seconds per seed. Smoke-test with --smoke before an array.
"""
from __future__ import annotations
import argparse, json, os, itertools
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from ibrg_llm.vib import VIB

BETAS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
CONDITIONS = ("aligned", "recomb")
EVAL_RHOS = (1.0, 0.66, 0.33, 0.0)          # shortcut-label corr per eval environment
SC_VAL = 4.0                                # one-hot shortcut cue magnitude


def make_data(rng, n, protos, n_classes, sigc, kind, rho=None):
    """kind: 'aligned' | 'recomb' | 'env' (needs rho=P(shortcut==y)). Returns X, y, shortcut."""
    n_c = protos.shape[1]
    y = rng.integers(0, n_classes, n)
    Xc = protos[y] + sigc * rng.standard_normal((n, n_c))
    if kind == "aligned":
        sc = y.copy()
    elif kind == "recomb":
        sc = y.copy(); half = rng.random(n) < 0.5
        sc[half] = rng.integers(0, n_classes, int(half.sum()))
    else:                                    # eval environment at correlation rho
        sc = y.copy(); r = rng.random(n) >= rho
        sc[r] = rng.integers(0, n_classes, int(r.sum()))
    Xs = np.zeros((n, n_classes), np.float32); Xs[np.arange(n), sc] = SC_VAL
    X = np.concatenate([Xc, Xs], axis=1).astype(np.float32)
    return X, y.astype(np.int64), sc.astype(np.int64)


def train_vib(beta, Xtr, ytr, n_classes, cfg, seed):
    torch.manual_seed(seed)
    m = VIB(Xtr.shape[1], cfg["z_dim"], n_classes)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(cfg["epochs"]):
        m.train(); opt.zero_grad()
        lg, kl, _ = m(xt)
        (F.cross_entropy(lg, yt) + beta * kl.mean()).backward(); opt.step()
    m.eval()
    return m


def mu_of(m, X):
    with torch.no_grad():
        return m(torch.tensor(X), sample=False)[2].numpy()


def build_direction_bank(m, protos, n_classes, sigc, rng, n_draw=24):
    """Ground-truth-labelled unit directions in z, one per ordered class pair (a,b).
    causal v_ab  = mean mu(causal=proto_a, shortcut=ref) - mean mu(causal=proto_b, shortcut=ref)
    spurious v_ab= mean mu(causal=ref, shortcut=onehot_a) - mean mu(causal=ref, shortcut=onehot_b)
    Returns list of dicts: {v, a, b, kind}."""
    n_c = protos.shape[1]
    ref_c = protos.mean(0, keepdims=True)                 # neutral causal input
    ref_sc = np.zeros((1, n_classes), np.float32)         # neutral (empty) shortcut input

    def enc_causal(cls):
        Xc = np.repeat(protos[cls:cls + 1], n_draw, 0) + sigc * rng.standard_normal((n_draw, n_c))
        return mu_of(m, np.concatenate([Xc, np.repeat(ref_sc, n_draw, 0)], 1).astype(np.float32)).mean(0)

    def enc_shortcut(cls):
        Xc = np.repeat(ref_c, n_draw, 0) + sigc * rng.standard_normal((n_draw, n_c))
        sc = np.zeros((n_draw, n_classes), np.float32); sc[:, cls] = SC_VAL
        return mu_of(m, np.concatenate([Xc, sc], 1).astype(np.float32)).mean(0)

    muc = [enc_causal(c) for c in range(n_classes)]
    mus = [enc_shortcut(c) for c in range(n_classes)]
    bank = []
    for a, b in itertools.permutations(range(n_classes), 2):
        for kind, reps in (("causal", muc), ("spurious", mus)):
            v = reps[a] - reps[b]; nv = np.linalg.norm(v)
            if nv > 1e-8:
                bank.append({"v": (v / nv).astype(np.float32), "a": a, "b": b, "kind": kind})
    return bank


def _pair_decode(proj, y, a, b):
    """Balanced acc of separating class a vs class b from a 1-D projection (2-fold)."""
    mask = (y == a) | (y == b)
    s = proj[mask].reshape(-1, 1); t = (y[mask] == a).astype(int)
    if t.sum() < 4 or (1 - t).sum() < 4:
        return np.nan
    h = len(s) // 2
    if len(np.unique(t[:h])) < 2:
        return np.nan
    pred = LogisticRegression(max_iter=300).fit(s[:h], t[:h]).predict(s[h:])
    return balanced_accuracy_score(t[h:], pred)


def foliation_scores(m, bank, Xtr, ytr, eval_envs):
    """Per direction: observational (train) decodability and cross-environment invariance
    (min over envs) of the a-vs-b split. Returns arrays aligned with bank + summary AUCs."""
    mu_tr = mu_of(m, Xtr)
    mu_envs = [(mu_of(m, Xe), ye) for (Xe, ye, _) in eval_envs]
    obs, inv, lab = [], [], []
    for d in bank:
        v, a, b = d["v"], d["a"], d["b"]
        obs.append(_pair_decode(mu_tr @ v, ytr, a, b))
        env_acc = [_pair_decode(mu_e @ v, ye, a, b) for (mu_e, ye) in mu_envs]
        inv.append(np.nanmin(env_acc))
        lab.append(1 if d["kind"] == "causal" else 0)
    obs, inv, lab = np.array(obs), np.array(inv), np.array(lab)
    ok = ~(np.isnan(obs) | np.isnan(inv))
    obs, inv, lab = obs[ok], inv[ok], lab[ok]

    def auc(x):
        return float(roc_auc_score(lab, x)) if len(np.unique(lab)) == 2 else float("nan")

    return {
        "obs_causal": float(obs[lab == 1].mean()), "obs_spurious": float(obs[lab == 0].mean()),
        "inv_causal": float(inv[lab == 1].mean()), "inv_spurious": float(inv[lab == 0].mean()),
        "obs_auc": auc(obs), "inv_auc": auc(inv),
        "separation": float(inv[lab == 1].mean() - inv[lab == 0].mean()),
        "n_dirs": int(len(lab)),
    }


def fdt_scores(m, bank, Xood, eps):
    """E3: susceptibility chi = mean |d logit_correct| under do(z += eps*v) over causal
    directions and OOD data; fluctuation = mean total variance of mu on OOD data."""
    mu = torch.tensor(mu_of(m, Xood))
    with torch.no_grad():
        base = m.cls(mu)
        resp = []
        for d in bank:
            if d["kind"] != "causal":
                continue
            vt = torch.tensor(d["v"])
            dlog = (m.cls(mu + eps * vt) - base)                 # change in all logits
            resp.append(dlog.abs().mean().item())                # mean |Δlogit| per unit kick
    chi = float(np.mean(resp)) if resp else float("nan")
    fluct = float(mu.numpy().var(axis=0).sum())                  # trace of Cov(mu)
    return {"chi": chi, "fluct_var_z": fluct}


def run_condition(condition, protos, n_classes, sigc, cfg, seed):
    rng = np.random.default_rng(seed + 1)
    Xtr, ytr, _ = make_data(rng, cfg["n_train"], protos, n_classes, sigc, condition)
    eval_envs = [make_data(rng, cfg["n_env"], protos, n_classes, sigc, "env", rho) for rho in EVAL_RHOS]
    Xood, yood, _ = make_data(rng, cfg["n_test"], protos, n_classes, sigc, "env", 1.0 / n_classes)
    sweep = []
    for beta in BETAS:
        m = train_vib(beta, Xtr, ytr, n_classes, cfg, seed)
        with torch.no_grad():
            lg_tr, kl_tr, _ = m(torch.tensor(Xtr), sample=False)
            lg_ood, _, mu_ood = m(torch.tensor(Xood), sample=False)
        train_acc = float((lg_tr.argmax(1).numpy() == ytr).mean())
        ood_acc = float((lg_ood.argmax(1).numpy() == yood).mean())
        # shortcut retention: decode the (OOD) shortcut class from z
        _, _, scood = make_data(np.random.default_rng(seed + 99), cfg["n_test"], protos,
                                n_classes, sigc, "env", 1.0 / n_classes)
        bank = build_direction_bank(m, protos, n_classes, sigc, rng)
        eps = 0.3 * float(np.median(np.linalg.norm(mu_ood.numpy() - mu_ood.numpy().mean(0), axis=1)))
        fol = foliation_scores(m, bank, Xtr, ytr, eval_envs)
        fdt = fdt_scores(m, bank, Xood, eps)
        row = {"beta": beta, "rate": float(kl_tr.mean()), "train_acc": train_acc,
               "ood_acc": ood_acc, "eps": eps, **fol, **fdt}
        sweep.append(row)
        print(f"[{condition}] b={beta:<6} R={row['rate']:6.2f} OOD={ood_acc:.3f} | "
              f"obs_auc={fol['obs_auc']:.2f}(c{fol['obs_causal']:.2f}/s{fol['obs_spurious']:.2f}) "
              f"inv_auc={fol['inv_auc']:.2f}(c{fol['inv_causal']:.2f}/s{fol['inv_spurious']:.2f}) "
              f"sep={fol['separation']:+.3f} chi={fdt['chi']:.3f} var={fdt['fluct_var_z']:.2f}", flush=True)
    bstar = max(sweep, key=lambda r: r["ood_acc"])["beta"]
    peak_sep = max(sweep, key=lambda r: (r["separation"] if r["separation"] == r["separation"] else -9))["beta"]
    print(f"[{condition}] beta* (OOD optimum)={bstar:g}  separation-peak beta={peak_sep:g}", flush=True)
    return sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--n_c", type=int, default=4)
    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--sigc", type=float, default=0.9)
    ap.add_argument("--proto_c", type=float, default=2.0)
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_env", type=int, default=1200)
    ap.add_argument("--n_test", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--smoke", action="store_true", help="tiny/fast config to shake out bugs")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_env, args.n_test, args.epochs = 400, 400, 800, 150
        global BETAS
        BETAS = [0.0, 1e-2, 1e-1, 1.0]
    tag = args.tag or (f"smoke" if args.smoke else f"seed{args.seed}")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    rng = np.random.default_rng(args.seed)
    protos = rng.standard_normal((args.n_classes, args.n_c)) * args.proto_c
    cfg = {"z_dim": args.z_dim, "epochs": args.epochs, "n_train": args.n_train,
           "n_env": args.n_env, "n_test": args.n_test}
    res = {c: run_condition(c, protos, args.n_classes, args.sigc, cfg, args.seed) for c in CONDITIONS}
    odir = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(odir, exist_ok=True)
    json.dump({"seed": args.seed, "cfg": vars(args), "betas": BETAS, "eval_rhos": list(EVAL_RHOS),
               "conditions": res}, open(os.path.join(odir, f"crit_{tag}.json"), "w"), indent=2)
    print(f"[saved] out/crit_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
