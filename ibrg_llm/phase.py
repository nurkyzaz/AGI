"""Experiment B (IB phase diagram) + D (selective preservation vs cue reliability).

B: shortcut_pref over a grid of (shortcut_strength x demos-per-class k) — the
   motivation's IB phase transition made concrete (does reliance flip sharply?).
D: at fixed k, probe COLOR (shortcut) and SHAPE (rule) decodability across depth
   as the shortcut strength varies — is an unreliable cue represented less /
   discarded, as selective coarse-graining predicts?
CPU-friendly (logit + hidden-state reads; no generation).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from ibrg_llm.shortcut_task import make_task, LABELS

STRENGTHS = [1.0, 0.9, 0.8, 0.7, 0.5]
KS = [1, 2, 3, 4]


def probe(X, y):
    ns = int(min(5, np.bincount(y)[np.bincount(y) > 0].min()))
    if ns < 2:
        return float("nan")
    a = [balanced_accuracy_score(y[te], make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[tr], y[tr]).predict(X[te]))
         for tr, te in StratifiedKFold(ns, shuffle=True, random_state=0).split(X, y)]
    return float(np.mean(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--n_probe", type=int, default=120)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"seed{args.seed}"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                 output_hidden_states=True).to("cpu").eval()
    sp_dig = {L: tok(f" {L}", add_special_tokens=False)["input_ids"] for L in LABELS}
    space_id = sp_dig[LABELS[0]][0]
    lab_id = {L: sp_dig[L][-1] for L in LABELS}
    assert len(set(lab_id.values())) == len(lab_id)

    def logits_hidden(prompt):
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        ids = torch.cat([ids, torch.tensor([[space_id]])], dim=1)
        with torch.no_grad():
            o = model(input_ids=ids)
        return o.logits[0, -1], o.hidden_states

    def behav(tasks):
        pr = []
        for t in tasks:
            lg, _ = logits_hidden(t.prompt)
            p = torch.softmax(torch.stack([lg[lab_id[L]] for L in LABELS]), 0)
            ps, prule = float(p[LABELS.index(t.shortcut_label)]), float(p[LABELS.index(t.rule_label)])
            pr.append(ps / (ps + prule + 1e-9))
        return float(np.mean(pr))

    rng = np.random.default_rng(args.seed)
    # ---- B: phase diagram ----
    grid = {}
    for s in STRENGTHS:
        for k in KS:
            tasks = [make_task(rng, k_per=k, conflict=True, shortcut_strength=s) for _ in range(args.n)]
            grid[f"s{s}_k{k}"] = behav(tasks)
            print(f"[B] strength={s} k={k} shortcut_pref={grid[f's{s}_k{k}']:.3f}", flush=True)

    # ---- D: cue decodability across depth vs strength (k=2) ----
    depth = {}
    for s in STRENGTHS:
        Hs, yc, ysh = [], [], []
        for _ in range(args.n_probe):
            t = make_task(rng, k_per=2, conflict=True, shortcut_strength=s)
            _, hs = logits_hidden(t.prompt)
            Hs.append(torch.stack([h[0, -1, :] for h in hs]).float().numpy())
            yc.append(t.color_class); ysh.append(t.shape_class)
        H = np.stack(Hs); yc = np.array(yc); ysh = np.array(ysh); L = H.shape[1]
        depth[f"s{s}"] = {"color": [probe(H[:, l, :], yc) for l in range(L)],
                          "shape": [probe(H[:, l, :], ysh) for l in range(L)]}
        print(f"[D] strength={s} best color_probe={np.nanmax(depth[f's{s}']['color']):.3f} "
              f"best shape_probe={np.nanmax(depth[f's{s}']['shape']):.3f}", flush=True)

    odir = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(odir, exist_ok=True)
    json.dump({"model": args.model, "seed": args.seed, "B_phase": grid, "D_depth": depth,
               "strengths": STRENGTHS, "ks": KS},
              open(os.path.join(odir, f"phase_{tag}.json"), "w"), indent=2)
    print(f"[saved] out/phase_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
