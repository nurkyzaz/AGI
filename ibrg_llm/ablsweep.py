"""Experiment C (causal steering of generalization).

C1 layer-sweep: ablate the COLOR (shortcut) subspace at EACH layer and measure
   the change in shortcut_pref -> which layer is load-bearing? (+ random-subspace
   control at that layer).
C3 dose-response: at the load-bearing layer, ADD +/- alpha * (top shortcut
   direction) and sweep alpha -> can we push the model toward the shortcut or the
   rule on demand? CPU-friendly (logit reads).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch
from ibrg_llm.shortcut_task import make_task, LABELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_abl", type=int, default=100)
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
    conflict = [make_task(rng, conflict=True) for _ in range(args.n_abl)]
    base = behav(conflict)

    # collect per-layer activations to build the color subspace at every layer
    Hs, yc = [], []
    for _ in range(args.n_probe):
        t = make_task(rng, conflict=True)
        _, hs = logits_hidden(t.prompt)
        Hs.append(torch.stack([h[0, -1, :] for h in hs]).float().numpy())
        yc.append(t.color_class)
    H = np.stack(Hs); yc = np.array(yc); L = H.shape[1]; d = H.shape[2]

    def color_subspace(layer):
        X = H[:, layer, :]
        mus = np.stack([X[yc == c].mean(0) for c in np.unique(yc)])
        M = mus - mus.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        return Vt[:len(np.unique(yc)) - 1]

    def hook_ablate(B, layer_idx):
        Bt = torch.tensor(B, dtype=torch.float32)
        def hook(m, i, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = h - (h @ Bt.T) @ Bt
            return ((h2,) + tuple(out[1:])) if isinstance(out, tuple) else h2
        return hook

    def hook_add(vec, alpha):
        v = torch.tensor(vec, dtype=torch.float32)
        def hook(m, i, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = h + alpha * v
            return ((h2,) + tuple(out[1:])) if isinstance(out, tuple) else h2
        return hook

    def run_with(layer_k, hook):
        mod = model.model.layers[layer_k - 1]
        hnd = mod.register_forward_hook(hook)
        try:
            return behav(conflict)
        finally:
            hnd.remove()

    # ---- C1: ablate color subspace at each layer ----
    c1 = {}
    for lk in range(1, L):
        c1[lk] = run_with(lk, hook_ablate(color_subspace(lk), lk))
        print(f"[C1] layer={lk} shortcut_pref={c1[lk]:.3f} (base {base:.3f}, effect {base-c1[lk]:+.3f})", flush=True)
    best_layer = max(c1, key=lambda k: base - c1[k])           # most reduction
    # random-subspace control at the best layer
    G = np.random.default_rng(args.seed).standard_normal((color_subspace(best_layer).shape[0], d))
    Q, _ = np.linalg.qr(G.T); rand_ctrl = run_with(best_layer, hook_ablate(Q.T[:G.shape[0]], best_layer))
    print(f"[C1] best_layer={best_layer} color_ablate={c1[best_layer]:.3f} random_ctrl={rand_ctrl:.3f}", flush=True)

    # ---- C3: dose-response steering at best layer ----
    dvec = color_subspace(best_layer)[0]                       # top shortcut axis
    scale = float(np.linalg.norm(H[:, best_layer, :], axis=1).mean())
    c3 = {}
    for a in [-3, -2, -1, 0, 1, 2, 3]:
        c3[a] = base if a == 0 else run_with(best_layer, hook_add(dvec, a * 0.15 * scale))
        print(f"[C3] alpha={a} shortcut_pref={c3[a]:.3f}", flush=True)

    odir = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(odir, exist_ok=True)
    json.dump({"model": args.model, "seed": args.seed, "base": base,
               "C1_layer_ablate": c1, "best_layer": best_layer, "C1_random_ctrl": rand_ctrl,
               "C3_dose": c3},
              open(os.path.join(odir, f"abl_{tag}.json"), "w"), indent=2)
    print(f"[saved] out/abl_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
