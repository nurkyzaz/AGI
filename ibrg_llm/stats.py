"""Every number used in the write-up, computed from ibrg_llm/out/*.json only.

Run:  python -m ibrg_llm.stats > ibrg_llm/out/stats.txt

Conventions: SD is the sample standard deviation over seeds (ddof=1);
SE = SD / sqrt(n); n is the number of seeds (independent runs) unless stated.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(__file__), "out")
MODELS = [("0.5B", "shortcut_q05b_s*.json"), ("1.5B", "shortcut_q15b_s*.json"),
          ("3B", "shortcut_seed*.json")]


def load(pat):
    files = sorted(glob.glob(os.path.join(OUT, pat)))
    return [json.load(open(f)) for f in files], [os.path.basename(f) for f in files]


def msd(v):
    v = np.asarray(v, float)
    n = len(v)
    sd = v.std(ddof=1) if n > 1 else float("nan")
    return v.mean(), sd, sd / np.sqrt(n), n


def fmt(v, d=3):
    m, sd, se, n = msd(v)
    return f"mean {m:.{d}f}  SD {sd:.{d}f}  SE {se:.{d}f}  (n={n})"


def r_p(x, y):
    r, p = stats.pearsonr(x, y)
    return f"r = {r:+.2f} (p = {p:.3f}, n = {len(x)})"


def header(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def track_a():
    runs = {}
    for name, pat in MODELS:
        rs, files = load(pat)
        runs[name] = rs
        print(f"{name}: {len(rs)} files: {', '.join(files)}")

    header("A1. Shortcut preference under conflict (0.5 = no preference; >0.5 = leans to shortcut)")
    print("n_behav = 160 conflict items per seed (runner.py default; all sbatch files use it)")
    sp = {k: [r["H1"]["shortcut_pref"] for r in v] for k, v in runs.items()}
    for k in sp:
        print(f"  {k:5s} shortcut_pref  {fmt(sp[k])}   per-seed: "
              + ", ".join(f"{x:.3f}" for x in sp[k]))
    for k, v in runs.items():
        print(f"  {k:5s} aligned_acc    {fmt([r['H1']['aligned_acc'] for r in v])}")
    for a, b in [("0.5B", "1.5B"), ("1.5B", "3B"), ("0.5B", "3B")]:
        t = stats.ttest_ind(sp[a], sp[b], equal_var=False)
        print(f"  Welch t-test {a} vs {b}: t = {t.statistic:.2f}, df = {t.df:.1f}, p = {t.pvalue:.4f}")
    for k in sp:
        t = stats.ttest_1samp(sp[k], 0.5)
        print(f"  one-sample t-test {k} vs 0.5: t = {t.statistic:.2f}, df = {len(sp[k]) - 1}, p = {t.pvalue:.4f}")

    header("A2. Removing the shortcut (color) directions at one fixed layer (H4)")
    print("effect = base - ablated shortcut_pref (positive = less shortcut use after removal)")
    for k, v in runs.items():
        c = np.array([r["H4"]["base"] - r["H4"]["color_ablate"] for r in v])
        rnd = np.array([r["H4"]["base"] - r["H4"]["random_ablate"] for r in v])
        layer = sorted({r["abl_layer"] for r in v})
        print(f"  {k} (ablation layer {layer}):")
        print(f"    shortcut-ablation effect  {fmt(c)}   per-seed: " + ", ".join(f"{x:+.3f}" for x in c))
        print(f"    random-ablation effect    {fmt(rnd)}   per-seed: " + ", ".join(f"{x:+.3f}" for x in rnd))
        t = stats.ttest_rel(c, rnd)
        print(f"    paired t-test shortcut vs random: t = {t.statistic:.2f}, df = {len(c) - 1}, p = {t.pvalue:.3f}"
              f"; seeds with shortcut effect > 0: {int((c > 0).sum())}/{len(c)}")

    header("A3. Layer sweep (abl_seed*.json, 3B)")
    ab, files = load("abl_seed*.json")
    print(f"files: {', '.join(files)}  (n = {len(ab)} seeds)")
    best_eff, rnd_eff = [], []
    for r in ab:
        L = {int(k): r["base"] - v for k, v in r["C1_layer_ablate"].items()}
        bl = r["best_layer"]
        best_eff.append(L[bl]); rnd_eff.append(r["base"] - r["C1_random_ctrl"])
        print(f"  seed {r['seed']}: best layer {bl} of {len(L)}; best-layer effect {L[bl]:+.3f}; "
              f"one random direction at that layer {rnd_eff[-1]:+.3f}")
    print(f"  best-layer effect   {fmt(best_eff)}")
    print(f"  random-ctrl effect  {fmt(rnd_eff)}")
    print("  NOTE: best layer is picked as the max over 36 layers, then compared with ONE random")
    print("  direction at that layer. Selecting the max inflates the shortcut effect (selection bias).")

    header("A4. Rule-vs-shortcut probe gap vs rule-following (E1)")
    print("x = max_layer(rule probe acc) - max_layer(shortcut probe acc); y = 1 - shortcut_pref")
    X, Y, Xs, G = [], [], [], []
    for k, v in runs.items():
        for r in v:
            h = r["H2_H3"]
            X.append(np.nanmax(h["rule_curve"]) - np.nanmax(h["shortcut_curve"]))
            Xs.append(np.nanmax(h["shortcut_curve"]))
            Y.append(1 - r["H1"]["shortcut_pref"]); G.append(k)
    X, Y, Xs, G = map(np.array, (X, Y, Xs, G))
    print(f"  pooled: {r_p(X, Y)}")
    for k in sp:
        m = G == k
        print(f"  within {k}: {r_p(X[m], Y[m])}")
    Xd = X - np.array([X[G == g].mean() for g in G]); Yd = Y - np.array([Y[G == g].mean() for g in G])
    r = np.corrcoef(Xd, Yd)[0, 1]
    print(f"  within-model pooled (model means removed): r = {r:+.2f} (n = 15, 3 means removed)")

    header("A5. Shortcut probe accuracy vs shortcut use (F1, 'monitor')")
    sp_all = 1 - Y
    print(f"  pooled: {r_p(Xs, sp_all)}")
    for k in sp:
        m = G == k
        print(f"  within {k}: {r_p(Xs[m], sp_all[m])}")
    Xd = Xs - np.array([Xs[G == g].mean() for g in G]); Yd = sp_all - np.array([sp_all[G == g].mean() for g in G])
    print(f"  within-model pooled (model means removed): r = {np.corrcoef(Xd, Yd)[0, 1]:+.2f}")


def sweep_table(rows, keys, betas):
    print("  " + "beta".ljust(8) + "".join(k.rjust(26) for k in keys))
    for i, b in enumerate(betas):
        cells = []
        for k in keys:
            m, sd, _, n = msd([r[i][k] for r in rows])
            cells.append(f"{m:.3f} ± {sd:.3f} SD".rjust(26))
        print("  " + f"{b:g}".ljust(8) + "".join(cells))


def track_b():
    header("B1. Boundary: cheap shortcut, compression vs intervention (boundary_seed*.json)")
    br, files = load("boundary_seed*.json")
    nc = br[0]["cfg"]["n_classes"]
    print(f"n = {len(br)} seeds; n_classes = {nc} -> chance = {1 / nc:.2f}")
    for cond in ("spurious", "intervention"):
        rows = [r["conditions"][cond] for r in br]
        betas = [s["beta"] for s in rows[0]]
        print(f" condition = {cond}")
        sweep_table(rows, ["train_acc", "test_acc", "shortcut_retention"], betas)

    header("B2. VIB: useless noise feature (vib_seed*.json)")
    vr, _ = load("vib_seed*.json")
    rows = [r["sweep"] for r in vr]
    betas = [s["beta"] for s in rows[0]]
    print(f"n = {len(vr)} seeds; chance = {1 / vr[0]['cfg']['n_classes']:.2f}")
    sweep_table(rows, ["test_acc", "nuisance_recon", "causal_retention"], betas)
    te = [np.mean([r[i]["test_acc"] for r in rows]) for i in range(len(betas))]
    print(f"  best mean test_acc {max(te):.3f} at beta {betas[int(np.argmax(te))]:g}; at beta 0: {te[0]:.3f}")

    header("B3. Rule vs shortcut directions (crit_seed*.json)")
    cr, _ = load("crit_seed*.json")
    print(f"n = {len(cr)} seeds")
    for cond in ("aligned", "recomb"):
        rows = [r["conditions"][cond] for r in cr]
        betas = cr[0]["betas"]
        print(f" condition = {cond}  (obs = read off training data; inv = predict across environments)")
        sweep_table(rows, ["obs_causal", "obs_spurious", "obs_auc"], betas)
        sweep_table(rows, ["inv_causal", "inv_spurious", "inv_auc"], betas)
        if cond == "recomb":
            sweep_table(rows, ["ood_acc"], betas)


if __name__ == "__main__":
    print("stats.py — all numbers from ibrg_llm/out/*.json (SD over seeds, ddof=1)")
    track_a()
    track_b()
