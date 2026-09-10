"""Step 0 / probe sweep — the go/no-go and the headline deliverable (GPU node).

Two questions, both must get an interpretable answer:

  (A) BEHAVIORAL: does the model engage with the serialized few-shot rule task at
      all — well-formed grids, beating the trivial copy-input / all-zero baselines?
  (B) REPRESENTATIONAL: is the *causal* rule latent linearly decodable from the
      residual stream before the answer, above a shuffled-label floor? And how
      does that compare, across depth, to a *nuisance* latent (`frame_color`)?
      The IB=RG prediction: causal decodability is built up / preserved with
      depth while surface nuisance is selectively discarded.

Generic over families: probes the family's FIRST causal variable (LabelEncoded)
vs `frame_color`. Per-job outputs are tagged so parallel family jobs don't clash.

Design notes: probe position = last token of the chat-formatted prompt (rule is
"decided" there); one forward per stimulus, keep only [layer,-1,:]; probe =
logistic regression, 5-fold stratified CV, balanced accuracy; label-shuffled
control = chance floor for the same data.
"""
from __future__ import annotations

import argparse
import re
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from ibrg_llm.stimuli import make_stimulus

GRID = 12


def parse_grid(text: str):
    rows = []
    for line in text.splitlines():
        nums = re.findall(r"-?\d+", line)
        if len(nums) >= GRID:
            rows.append([int(x) for x in nums[:GRID]])
        if len(rows) == GRID:
            break
    return np.array(rows) if len(rows) == GRID else None


def cell_acc(pred, tgt):
    return 0.0 if pred is None else float((pred == tgt).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--family", default="F3_reflect")
    ap.add_argument("--tag", default=None, help="output tag (default = family)")
    ap.add_argument("--n_probe", type=int, default=300)
    ap.add_argument("--n_behav", type=int, default=40)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()
    tag = args.tag or args.family

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[step0] loading {args.model} on {args.device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="cuda",
            output_hidden_states=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, output_hidden_states=True).to("cpu")
    model.eval()
    dev = next(model.parameters()).device
    print(f"[step0] device={dev} n_layers={model.config.num_hidden_layers} family={args.family}", flush=True)

    def fmt(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)

    # ---------- (A) behavioral ----------
    rng = np.random.default_rng(args.seed)
    ex_hits = cells = copy_cells = zero_cells = wellformed = 0
    for _ in range(args.n_behav):
        s = make_stimulus(args.family, rng, k=args.k)
        ids = tok(fmt(s.prompt), return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=200, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_grid(gen)
        tgt = np.array([[int(x) for x in r.split()] for r in s.target_text.splitlines()])
        qin = np.array([[int(x) for x in r.split()]
                        for r in s.prompt.split("Input:\n")[-1].split("\nOutput")[0].splitlines()])
        wellformed += pred is not None
        ex_hits += int(pred is not None and np.array_equal(pred, tgt))
        cells += cell_acc(pred, tgt)
        copy_cells += float((qin == tgt).mean())
        zero_cells += float((tgt == 0).mean())
    n = args.n_behav
    print("\n===== (A) BEHAVIORAL =====")
    print(f"exact-match={ex_hits/n:.3f} cell_model={cells/n:.3f} "
          f"cell_copy={copy_cells/n:.3f} cell_zero={zero_cells/n:.3f} wellformed={wellformed/n:.3f}")

    # ---------- (B) probe first causal var vs frame_color across layers ----------
    rng = np.random.default_rng(args.seed + 1)
    H, yc_raw, yu = [], [], []
    cname = None
    for i in range(args.n_probe):
        s = make_stimulus(args.family, rng, k=args.k)
        if cname is None:
            cname = list(s.c_labels)[0]
        ids = tok(fmt(s.prompt), return_tensors="pt").to(dev)
        with torch.no_grad():
            hs = model(**ids).hidden_states           # (L+1) x (1,seq,hid)
        H.append(torch.stack([h[0, -1, :] for h in hs]).float().cpu().numpy())
        yc_raw.append(str(s.c_labels[cname]))
        yu.append(int(s.u_labels["frame_color"]))
        if (i + 1) % 50 == 0:
            print(f"[probe] {i+1}/{args.n_probe}", flush=True)
    H = np.stack(H)
    y_c = LabelEncoder().fit_transform(yc_raw)
    y_u = np.array(yu)
    c_chance = 1.0 / len(set(y_c)); u_chance = 1.0 / len(set(y_u))

    def probe_layer(X, y):
        counts = np.bincount(y)
        nsplits = int(min(5, counts[counts > 0].min()))  # robust to rare classes / small N
        if nsplits < 2:
            return float("nan")
        accs = []
        for tr, te in StratifiedKFold(nsplits, shuffle=True, random_state=0).split(X, y):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
        return float(np.mean(accs))

    L = H.shape[1]
    print(f"\n===== (B) PROBES  causal={cname}(chance {c_chance:.2f}) "
          f"nuisance=frame_color(chance {u_chance:.2f}) =====")
    print(f"{'layer':>5} {'causal':>7} {'c_shuf':>7} {'nuis':>7} {'n_shuf':>7}")
    res = {"layer": [], "causal": [], "causal_shuf": [], "nuis": [], "nuis_shuf": []}
    for Li in range(L):
        X = H[:, Li, :]
        c = probe_layer(X, y_c)
        c_sh = probe_layer(X, np.random.default_rng(0).permutation(y_c))
        u = probe_layer(X, y_u)
        u_sh = probe_layer(X, np.random.default_rng(0).permutation(y_u))
        for kk, vv in zip(res, (Li, c, c_sh, u, u_sh)):
            res[kk].append(float(vv))
        print(f"{Li:>5} {c:>7.3f} {c_sh:>7.3f} {u:>7.3f} {u_sh:>7.3f}")
    best_c = max(res["causal"])

    import json, os
    outdir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(outdir, exist_ok=True)
    summary = {
        "model": args.model, "family": args.family, "causal_var": cname, "k": args.k,
        "n_probe": args.n_probe, "n_behav": args.n_behav,
        "behavioral": {"exact": ex_hits / n, "cell_model": cells / n,
                       "cell_copy_input": copy_cells / n, "cell_all_zero": zero_cells / n,
                       "wellformed": wellformed / n},
        "probes": res, "best_causal": best_c,
        "chance": {"causal": c_chance, "nuisance": u_chance},
    }
    with open(os.path.join(outdir, f"step0_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [l / (L - 1) for l in res["layer"]]
        plt.figure(figsize=(7, 4.5))
        plt.plot(xs, res["causal"], "-o", ms=3, label=f"causal ({cname}, inferred)")
        plt.plot(xs, res["nuis"], "-s", ms=3, label="nuisance (frame_color, surface)")
        plt.plot(xs, res["causal_shuf"], "--", color="gray", lw=1, label="causal shuffled floor")
        plt.plot(xs, res["nuis_shuf"], ":", color="gray", lw=1, label="nuisance shuffled floor")
        plt.xlabel("normalized depth (layer / final)"); plt.ylabel("probe balanced accuracy")
        plt.title(f"Causal vs nuisance decodability across depth\n{args.model} · {args.family}")
        plt.ylim(0, 1.02); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"step0_{tag}.png"), dpi=130)
        print(f"[saved] out/step0_{tag}.png + step0_{tag}.json")
    except Exception as e:
        print(f"[plot skipped] {e}")

    print("\n===== GO / NO-GO =====")
    beh = "engages" if (cells / n) > max(copy_cells, zero_cells) / n + 0.02 else "WEAK"
    rep = "decodable" if best_c > c_chance + 0.15 else "WEAK"
    print(f"behavioral: {beh} | causal decodable: {rep} (best {cname} probe {best_c:.3f} vs chance {c_chance:.2f})")
    print("GO" if rep == "decodable" else "REASSESS (see PLAN.md fallbacks)")


if __name__ == "__main__":
    main()
