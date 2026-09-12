"""Post-hoc aggregation + figures across all seed/model runs (E + F + summaries).

Reads ibrg_llm/out/*.json and produces one figure per experiment plus the
headline E1 scatter (selectivity predicts generalization). Runs on CPU in
seconds — invoke after the arrays finish:  python -m ibrg_llm.aggregate
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "out")


def load(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, pat)))]


def mean_ci(vals):
    v = np.array([x for x in vals if x == x])
    if len(v) == 0:
        return float("nan"), float("nan")
    boot = [np.random.default_rng(i).choice(v, len(v)).mean() for i in range(2000)]
    return float(v.mean()), float(np.std(boot))


def fig_base():
    runs = load("shortcut_seed*.json")
    if not runs:
        return
    sp = [r["H1"]["shortcut_pref"] for r in runs]
    al = [r["H1"]["aligned_acc"] for r in runs]
    eff_c = [r["H4"]["base"] - r["H4"]["color_ablate"] for r in runs]
    eff_r = [r["H4"]["base"] - r["H4"]["random_ablate"] for r in runs]
    m_sp, e_sp = mean_ci(sp); m_al, e_al = mean_ci(al)
    m_c, e_c = mean_ci(eff_c); m_r, e_r = mean_ci(eff_r)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(["shortcut_pref\n(conflict)", "aligned_acc\n(sanity)"], [m_sp, m_al],
              yerr=[e_sp, e_al], color=["C1", "C0"], capsize=4)
    ax[0].axhline(0.5, ls=":", color="gray"); ax[0].set_ylim(0, 1); ax[0].set_title(f"H1 behavioral (n={len(sp)} seeds)")
    ax[1].bar(["color-ablate", "random-ablate"], [m_c, m_r], yerr=[e_c, e_r],
              color=["C3", "gray"], capsize=4)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_title("H4 shortcut-ablation effect (Δ shortcut_pref)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_base_H1_H4.png"), dpi=130); plt.close()
    print("fig_base_H1_H4.png:", f"shortcut_pref={m_sp:.3f}±{e_sp:.3f} aligned={m_al:.3f} "
          f"color_eff={m_c:+.3f}±{e_c:.3f} random_eff={m_r:+.3f}±{e_r:.3f}")


def fig_scale():
    fig_pts = {}
    for abbr, pat in [("0.5B", "shortcut_q05b_s*.json"), ("1.5B", "shortcut_q15b_s*.json"),
                      ("3B", "shortcut_seed*.json")]:
        runs = load(pat)
        if runs:
            fig_pts[abbr] = mean_ci([r["H1"]["shortcut_pref"] for r in runs])
    if len(fig_pts) < 2:
        return
    xs = list(fig_pts); ys = [fig_pts[k][0] for k in xs]; es = [fig_pts[k][1] for k in xs]
    plt.figure(figsize=(6, 4)); plt.errorbar(xs, ys, yerr=es, marker="o", capsize=4)
    plt.axhline(0.5, ls=":", color="gray"); plt.ylim(0, 1)
    plt.ylabel("shortcut_pref (conflict)"); plt.title("A1 shortcut-reliance vs model scale")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_scale.png"), dpi=130); plt.close()
    print("fig_scale.png:", {k: round(v[0], 3) for k, v in fig_pts.items()})


def fig_phase():
    runs = load("phase_seed*.json")
    if not runs:
        return
    S = runs[0]["strengths"]; K = runs[0]["ks"]
    M = np.zeros((len(S), len(K)))
    for i, s in enumerate(S):
        for j, k in enumerate(K):
            M[i, j] = np.mean([r["B_phase"][f"s{s}_k{k}"] for r in runs])
    plt.figure(figsize=(6, 4.5)); im = plt.imshow(M, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
    plt.colorbar(im, label="shortcut_pref"); plt.xticks(range(len(K)), K); plt.yticks(range(len(S)), S)
    plt.xlabel("demos per class k"); plt.ylabel("shortcut strength")
    plt.title("B IB phase diagram (shortcut reliance)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_phase.png"), dpi=130); plt.close()
    print("fig_phase.png saved")


def fig_ablsweep():
    runs = load("abl_seed*.json")
    if not runs:
        return
    layers = sorted(int(k) for k in runs[0]["C1_layer_ablate"])
    curve = [np.mean([r["base"] - r["C1_layer_ablate"][str(l)] for r in runs]) for l in layers]
    # C3 is a perturbation-SENSITIVITY test (the color-subspace axis has arbitrary
    # sign, so we plot |shortcut_pref − base| vs |alpha|, NOT signed rule/shortcut steering).
    alphas = sorted(int(a) for a in runs[0]["C3_dose"])
    dose = [np.mean([abs(r["C3_dose"][str(a)] - r["base"]) for r in runs]) for a in alphas]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(layers, curve, "-o", ms=3); ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("Δ shortcut_pref (base − ablated)")
    ax[0].set_title("C1 causal layer-sweep (shortcut-subspace ablation)")
    ax[1].plot(alphas, dose, "-o", ms=3)
    ax[1].set_xlabel("perturbation magnitude α (arbitrary sign)")
    ax[1].set_ylabel("|Δ shortcut_pref| vs base")
    ax[1].set_title("C3 sensitivity to shortcut-subspace perturbation")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_ablsweep.png"), dpi=130); plt.close()
    print("fig_ablsweep.png saved")


def _model_runs():
    """All per-model×seed runs (base 3B + scale sweep), each with H1 + H2_H3."""
    tagged = [("3B", "shortcut_seed*.json"), ("1.5B", "shortcut_q15b_s*.json"),
              ("0.5B", "shortcut_q05b_s*.json")]
    out = []
    for name, pat in tagged:
        for r in load(pat):
            h = r.get("H2_H3") or r.get("H2")
            if h is None:
                continue
            out.append((name, np.nanmax(h["rule_curve"]) - np.nanmax(h["shortcut_curve"]),
                        np.nanmax(h["shortcut_curve"]), r["H1"]["shortcut_pref"]))
    return out


def fig_selectivity():
    """E1 (non-circular): across MODELS x SEEDS, does rule-vs-shortcut probe
    selectivity predict rule-following? Variation comes from model/seed, not from
    a knob that mechanically drives both quantities."""
    runs = _model_runs()
    if len(runs) < 3:
        return
    names = [r[0] for r in runs]
    xs = np.array([r[1] for r in runs]); ys = np.array([1 - r[3] for r in runs])
    rho = float(np.corrcoef(xs, ys)[0, 1])
    plt.figure(figsize=(6, 4.5))
    for nm in set(names):
        m = [i for i, n in enumerate(names) if n == nm]
        plt.scatter(xs[m], ys[m], label=nm, alpha=0.75)
    plt.xlabel("selectivity: rule − shortcut probe (best layer)")
    plt.ylabel("generalization: rule-following (1 − shortcut_pref)")
    plt.title(f"E1 selectivity vs generalization across models×seeds (r={rho:.2f})")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_selectivity.png"), dpi=130); plt.close()
    print(f"fig_selectivity.png: r={rho:.3f} over {len(xs)} model×seed runs")


def fig_monitor():
    """F1: is a probe of the shortcut cue a usable monitor — does higher
    shortcut-cue decodability go with more shortcut reliance?"""
    runs = _model_runs()
    if len(runs) < 3:
        return
    xs = np.array([r[2] for r in runs]); ys = np.array([r[3] for r in runs])
    rho = float(np.corrcoef(xs, ys)[0, 1])
    plt.figure(figsize=(6, 4.5)); plt.scatter(xs, ys, alpha=0.75, color="C3")
    plt.xlabel("shortcut-cue decodability (best layer)"); plt.ylabel("shortcut_pref")
    plt.title(f"F1 shortcut-probe as a monitor (r={rho:.2f})")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_monitor.png"), dpi=130); plt.close()
    print(f"fig_monitor.png: r={rho:.3f}")


def fig_vib():
    """Track B: the controlled IB=RG beta-sweep."""
    runs = load("vib_seed*.json")
    if not runs:
        return
    betas = [s["beta"] for s in runs[0]["sweep"]]
    avg = lambda k: [float(np.mean([r["sweep"][i][k] for r in runs])) for i in range(len(betas))]
    R, tr, te = avg("rate"), avg("train_acc"), avg("test_acc")
    nr, cr = avg("nuisance_recon"), avg("causal_retention")
    x = [max(b, 1e-5) for b in betas]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(x, te, "-o", label="test acc (OOD generalization)")
    ax[0].plot(x, tr, "-o", alpha=0.4, label="train acc")
    ax[0].plot(x, nr, "-s", label="nuisance reconstruction from z")
    ax[0].plot(x, cr, "-^", label="causal retention in z")
    ax[0].set_xscale("log"); ax[0].set_xlabel("β (compression)")
    ax[0].set_ylabel("accuracy / decodability"); ax[0].legend(fontsize=8)
    ax[0].set_title("Track B: IB=RG vs compression β")
    ax[1].plot(R, te, "-o")
    for i, b in enumerate(betas):
        ax[1].annotate(f"{b:g}", (R[i], te[i]), fontsize=6)
    ax[1].set_xlabel("rate R = KL(z) [nats]"); ax[1].set_ylabel("OOD test acc")
    ax[1].set_title("rate–distortion: rate vs generalization")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_vib.png"), dpi=130); plt.close()
    bi = int(np.argmax(te))
    print(f"fig_vib.png: best OOD test_acc={te[bi]:.3f} at β={betas[bi]:g} vs {te[0]:.3f} at β=0; "
          f"nuisance_recon {nr[0]:.3f}→{nr[bi]:.3f}; causal_retention={cr[bi]:.3f}")


def fig_boundary():
    """Track B boundary: compression can't fix a cheap shortcut; intervention can."""
    runs = load("boundary_seed*.json")
    if not runs:
        return
    betas = [r["beta"] for r in runs[0]["conditions"]["spurious"]]
    avg = lambda c, k: [float(np.mean([r["conditions"][c][i][k] for r in runs])) for i in range(len(betas))]
    nc = runs[0]["cfg"].get("n_classes", 4)
    x = [max(b, 1e-4) for b in betas]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(x, avg("spurious", "test_acc"), "-o", label="spurious (compression only)")
    ax[0].plot(x, avg("intervention", "test_acc"), "-o", label="+ intervention do(U)")
    ax[0].axhline(1.0 / nc, ls=":", color="gray", label="chance")
    ax[0].set_xscale("log"); ax[0].set_xlabel("β"); ax[0].set_ylabel("OOD test acc")
    ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
    ax[0].set_title("Compression can't fix the shortcut — intervention does")
    ax[1].plot(x, avg("spurious", "shortcut_retention"), "-s", label="spurious")
    ax[1].plot(x, avg("intervention", "shortcut_retention"), "-s", label="intervention")
    ax[1].set_xscale("log"); ax[1].set_xlabel("β"); ax[1].set_ylabel("shortcut retention in z")
    ax[1].legend(fontsize=8); ax[1].set_title("shortcut kept in z regardless of β")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_boundary.png"), dpi=130); plt.close()
    sp, iv = max(avg("spurious", "test_acc")), max(avg("intervention", "test_acc"))
    print(f"fig_boundary.png: best OOD test — spurious(compression)={sp:.3f} vs intervention={iv:.3f}")


if __name__ == "__main__":
    for f in (fig_base, fig_scale, fig_phase, fig_ablsweep, fig_selectivity, fig_monitor, fig_vib, fig_boundary):
        try:
            f()
        except Exception as e:
            print(f"[skip] {f.__name__}: {e}")
    print("done — figures in ibrg_llm/out/")
