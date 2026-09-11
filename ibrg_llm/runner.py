"""Shortcut-vs-rule experiment on a real LLM (CPU-friendly; one seed per run).

H1 behavioral: on CONFLICT queries, does the model prefer the shortcut (color) or
   the rule (shape) label? Read from next-token logits over the label tokens.
   shortcut_pref = p(shortcut)/(p(shortcut)+p(rule)), plus a hard argmax vote.
   Sanity: on ALIGNED queries accuracy should be high (model learned the mapping).
H2 probe: are shape-class (rule cue) and color-class (shortcut cue) linearly
   decodable from the residual stream across depth?
H4 causal: ablate the COLOR (shortcut) subspace from a layer and re-measure
   shortcut_pref, vs a random subspace of equal rank (control) and vs ablating the
   SHAPE (rule) subspace (contrast). If removing the shortcut subspace shifts the
   model toward the rule more than random, we causally steered generalization.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from ibrg_llm.shortcut_task import make_task, LABELS


def probe(X, y):
    ns = int(min(5, np.bincount(y)[np.bincount(y) > 0].min()))
    if ns < 2:
        return float("nan")
    accs = []
    for tr, te in StratifiedKFold(ns, shuffle=True, random_state=0).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        clf.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_behav", type=int, default=160)
    ap.add_argument("--n_aligned", type=int, default=80)
    ap.add_argument("--n_probe", type=int, default=160)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"seed{args.seed}"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[runner] load {args.model} on {args.device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    dt = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dt, output_hidden_states=True)
    model = model.to(args.device).eval()
    dev = args.device
    # " 1" tokenizes as [space, digit]; we want the DIGIT id, and we score it by
    # appending the shared space token so next-token logits are over the digit.
    sp_dig = {L: tok(f" {L}", add_special_tokens=False)["input_ids"] for L in LABELS}
    space_id = sp_dig[LABELS[0]][0]
    lab_id = {L: sp_dig[L][-1] for L in LABELS}
    print("[runner] space_id:", space_id, "label ids:", lab_id, flush=True)
    assert len(set(lab_id.values())) == len(lab_id), f"label token ids collide: {lab_id}"

    def last_logits_and_hidden(prompt):
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        ids = torch.cat([ids, torch.tensor([[space_id]])], dim=1).to(dev)  # score digit after "-> "
        with torch.no_grad():
            out = model(input_ids=ids)
        return out.logits[0, -1], out.hidden_states

    # diagnostic: does the model actually predict the right digit on an aligned prompt?
    _dt = make_task(np.random.default_rng(12345), conflict=False)
    _lg, _ = last_logits_and_hidden(_dt.prompt)
    _top = torch.topk(_lg, 5).indices.tolist()
    print(f"[diag] aligned correct={_dt.rule_label} model_top5={[tok.decode([i]) for i in _top]}", flush=True)

    # ---------- H1 behavioral ----------
    rng = np.random.default_rng(args.seed)
    conflict = [make_task(rng, conflict=True) for _ in range(args.n_behav)]
    aligned = [make_task(rng, conflict=False) for _ in range(args.n_aligned)]

    def behav(tasks):
        prefs, hard = [], []
        for t in tasks:
            lg, _ = last_logits_and_hidden(t.prompt)
            pr = torch.softmax(torch.stack([lg[lab_id[L]] for L in LABELS]), 0).cpu().numpy()
            p = {L: pr[i] for i, L in enumerate(LABELS)}
            ps, prule = p[t.shortcut_label], p[t.rule_label]
            prefs.append(float(ps / (ps + prule + 1e-9)))
            hard.append(int(ps > prule))
        return float(np.mean(prefs)), float(np.mean(hard))

    sc_pref, sc_hard = behav(conflict)
    al_pref, _ = behav(aligned)
    # aligned accuracy: does argmax label == the (agreeing) label?
    al_acc = []
    for t in aligned:
        lg, _ = last_logits_and_hidden(t.prompt)
        pick = max(LABELS, key=lambda L: float(lg[lab_id[L]]))
        al_acc.append(int(pick == t.rule_label))
    al_acc = float(np.mean(al_acc))
    print(f"\n[H1] conflict shortcut_pref={sc_pref:.3f} hard_vote_shortcut={sc_hard:.3f} "
          f"| aligned acc={al_acc:.3f} (sanity: model learned the mapping)", flush=True)

    # ---------- H2 probe + collect activations for ablation ----------
    Hs, y_shape, y_color, meta = [], [], [], []
    for i in range(args.n_probe):
        t = make_task(rng, conflict=True)
        _, hs = last_logits_and_hidden(t.prompt)
        Hs.append(torch.stack([h[0, -1, :] for h in hs]).float().cpu().numpy())
        y_shape.append(t.shape_class); y_color.append(t.color_class); meta.append(t)
    H = np.stack(Hs)                    # (N, L+1, d)
    y_shape = np.array(y_shape); y_color = np.array(y_color)
    L = H.shape[1]
    rule_curve = [probe(H[:, l, :], y_shape) for l in range(L)]
    short_curve = [probe(H[:, l, :], y_color) for l in range(L)]
    print(f"[H2] best rule(shape) probe={np.nanmax(rule_curve):.3f} "
          f"best shortcut(color) probe={np.nanmax(short_curve):.3f} (chance 0.333)", flush=True)

    # ---------- H4 ablation ----------
    # subspace of a cue = span of class-mean differences at a chosen layer
    def cue_subspace(layer, y):
        X = H[:, layer, :]
        mus = np.stack([X[y == c].mean(0) for c in np.unique(y)])
        M = mus - mus.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        r = len(np.unique(y)) - 1
        return Vt[:r]                   # (r, d) orthonormal rows

    def rand_subspace(r, d, seed):
        G = np.random.default_rng(seed).standard_normal((r, d))
        Q, _ = np.linalg.qr(G.T)
        return Q.T[:r]

    abl_layer = L // 2                  # mid-depth
    B_color = cue_subspace(abl_layer, y_color)
    B_shape = cue_subspace(abl_layer, y_shape)
    B_rand = rand_subspace(B_color.shape[0], H.shape[2], args.seed)

    def make_hook(B):
        Bt = torch.tensor(B, dtype=next(model.parameters()).dtype, device=dev)  # (r,d)
        def hook(mod, inp, out):
            is_tuple = isinstance(out, tuple)          # Qwen2 layer output form varies by version
            h = out[0] if is_tuple else out
            h2 = h - (h @ Bt.T) @ Bt                   # project the cue subspace out
            return ((h2,) + tuple(out[1:])) if is_tuple else h2
        return hook

    layer_mod = model.model.layers[abl_layer - 1]   # produces hidden_states[abl_layer]

    def behav_conflict_with(B):
        hnd = layer_mod.register_forward_hook(make_hook(B))
        try:
            pref, hard = behav(conflict)
        finally:
            hnd.remove()
        return pref, hard

    pref_color, hard_color = behav_conflict_with(B_color)
    pref_rand, hard_rand = behav_conflict_with(B_rand)
    pref_shape, hard_shape = behav_conflict_with(B_shape)
    print(f"[H4] ablate@layer{abl_layer} shortcut_pref: base={sc_pref:.3f} "
          f"color_ablate={pref_color:.3f} random={pref_rand:.3f} shape_ablate={pref_shape:.3f}", flush=True)
    print(f"     -> color-ablation effect={sc_pref-pref_color:+.3f}  random effect={sc_pref-pref_rand:+.3f}", flush=True)

    # ---------- H5 IB-transition: shortcut_pref vs demos-per-class k ----------
    ksweep = {}
    for kp in (1, 2, 3, 4):
        tk = [make_task(rng, k_per=kp, conflict=True) for _ in range(args.n_behav)]
        ksweep[kp], _ = behav(tk)
        print(f"[H5] k_per={kp} shortcut_pref={ksweep[kp]:.3f}", flush=True)

    out = {
        "model": args.model, "seed": args.seed, "abl_layer": abl_layer,
        "H1": {"shortcut_pref": sc_pref, "hard_vote_shortcut": sc_hard,
               "aligned_acc": al_acc, "aligned_pref": al_pref},
        "H2_H3": {"rule_curve": rule_curve, "shortcut_curve": short_curve, "chance": 1/3},
        "H4": {"base": sc_pref, "color_ablate": pref_color, "random_ablate": pref_rand,
               "shape_ablate": pref_shape},
        "H5_ksweep": ksweep,
    }
    odir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(odir, exist_ok=True)
    json.dump(out, open(os.path.join(odir, f"shortcut_{tag}.json"), "w"), indent=2)
    print(f"[saved] out/shortcut_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
