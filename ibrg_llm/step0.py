"""Step 0 — the go/no-go for the whole project (run on a GPU node).

Two questions, both must have an interpretable answer before we build anything:

  (A) BEHAVIORAL: does the model engage with the serialized few-shot rule task at
      all — i.e. does it produce well-formed grids and beat the trivial
      "copy the input" / "all background" baselines under k demos?
  (B) REPRESENTATIONAL: is the *causal* rule latent (F3's reflection `axis`,
      binary) linearly decodable from the residual stream before the answer,
      above a shuffled-label floor? And, for contrast, the *nuisance* latent
      (`frame_color`)?

Design choices (documented so they're auditable):
  * Probe position = the LAST token of the chat-formatted prompt (right before the
    model answers): where the rule must already be "decided". One forward pass per
    stimulus with output_hidden_states; we keep only [layer, -1, :] and drop the
    rest, so memory stays tiny.
  * Probe = logistic regression, 5-fold stratified CV, balanced accuracy; a
    label-shuffled control gives the chance floor for the SAME probe/data.
  * Family = F3_reflect: `axis` is binary (chance 0.5), the cleanest probe target.
    Clean arm only here (no shortcut barcode) — the shortcut/`do(U)` arm is later.
"""
from __future__ import annotations

import argparse
import re
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from ibrg_llm.stimuli import make_stimulus

GRID = 12


def parse_grid(text: str):
    """Best-effort: first GRID rows of GRID ints in the model's completion."""
    rows = []
    for line in text.splitlines():
        nums = re.findall(r"-?\d+", line)
        if len(nums) >= GRID:
            rows.append([int(x) for x in nums[:GRID]])
        if len(rows) == GRID:
            break
    if len(rows) != GRID:
        return None
    return np.array(rows)


def cell_acc(pred, tgt):
    if pred is None:
        return 0.0
    return float((pred == tgt).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--family", default="F3_reflect")
    ap.add_argument("--n_probe", type=int, default=300)   # stimuli for probing
    ap.add_argument("--n_behav", type=int, default=40)    # stimuli for generation
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[step0] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda", output_hidden_states=True
    )
    model.eval()
    dev = next(model.parameters()).device
    print(f"[step0] device={dev} n_layers={model.config.num_hidden_layers}", flush=True)

    def fmt(prompt: str):
        msgs = [{"role": "user", "content": prompt}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ---------- (A) behavioral: does it do the task? ----------
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
        tgt = np.array([[int(x) for x in row.split()] for row in s.target_text.splitlines()])
        qin = np.array([[int(x) for x in row.split()]
                        for row in s.prompt.split("Input:\n")[-1].split("\nOutput")[0].splitlines()])
        wellformed += pred is not None
        ex_hits += int(pred is not None and np.array_equal(pred, tgt))
        cells += cell_acc(pred, tgt)
        copy_cells += float((qin == tgt).mean())          # baseline: output == input
        zero_cells += float((tgt == 0).mean())            # baseline: all background
    n = args.n_behav
    print("\n===== (A) BEHAVIORAL =====")
    print(f"exact-match          : {ex_hits/n:.3f}")
    print(f"cell-acc (model)     : {cells/n:.3f}")
    print(f"cell-acc (copy-input): {copy_cells/n:.3f}   <- trivial baseline")
    print(f"cell-acc (all-zero)  : {zero_cells/n:.3f}   <- trivial baseline")
    print(f"well-formed grids    : {wellformed/n:.3f}")

    # ---------- (B) representational: probe C (axis) and U (frame) ----------
    rng = np.random.default_rng(args.seed + 1)
    H = []           # per-stimulus (n_layers+1, hidden)
    y_axis, y_frame = [], []
    for i in range(args.n_probe):
        s = make_stimulus(args.family, rng, k=args.k)
        ids = tok(fmt(s.prompt), return_tensors="pt").to(dev)
        with torch.no_grad():
            hs = model(**ids).hidden_states           # tuple (L+1) of (1,seq,hid)
        vec = torch.stack([h[0, -1, :] for h in hs]).float().cpu().numpy()
        H.append(vec)
        y_axis.append(0 if s.c_labels["axis"] == "horizontal" else 1)
        y_frame.append(int(s.u_labels["frame_color"]))
        if (i + 1) % 50 == 0:
            print(f"[probe] {i+1}/{args.n_probe}", flush=True)
    H = np.stack(H)                                    # (N, L+1, hid)
    y_axis = np.array(y_axis); y_frame = np.array(y_frame)

    def probe_layer(X, y, seed=0):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        accs = []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
        return float(np.mean(accs))

    L = H.shape[1]
    print("\n===== (B) PROBES  (balanced acc; axis chance=0.5, frame chance~0.125) =====")
    print(f"{'layer':>5} {'axis':>6} {'axis_shuf':>10} {'frame':>7} {'frame_shuf':>11}")
    res = {"layer": [], "axis": [], "axis_shuf": [], "frame": [], "frame_shuf": []}
    for L_i in range(L):
        X = H[:, L_i, :]
        a = probe_layer(X, y_axis)
        a_sh = probe_layer(X, np.random.default_rng(0).permutation(y_axis))
        f = probe_layer(X, y_frame)
        f_sh = probe_layer(X, np.random.default_rng(0).permutation(y_frame))
        for kk, vv in zip(res, (L_i, a, a_sh, f, f_sh)):
            res[kk].append(float(vv))
        print(f"{L_i:>5} {a:>6.3f} {a_sh:>10.3f} {f:>7.3f} {f_sh:>11.3f}")
    best_axis = max(res["axis"])

    # --- persist results + a normalized-depth selectivity plot (the H2 headline) ---
    import json, os
    outdir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(outdir, exist_ok=True)
    summary = {
        "model": args.model, "family": args.family, "k": args.k,
        "n_probe": args.n_probe, "n_behav": args.n_behav,
        "behavioral": {"exact": ex_hits / n, "cell_model": cells / n,
                       "cell_copy_input": copy_cells / n, "cell_all_zero": zero_cells / n,
                       "wellformed": wellformed / n},
        "probes": res, "best_axis": best_axis,
    }
    with open(os.path.join(outdir, "step0_results.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [l / (L - 1) for l in res["layer"]]
        plt.figure(figsize=(7, 4.5))
        plt.plot(xs, res["axis"], "-o", ms=3, label="axis (causal, inferred)")
        plt.plot(xs, res["frame"], "-s", ms=3, label="frame_color (nuisance, surface)")
        plt.plot(xs, res["axis_shuf"], "--", color="gray", lw=1, label="axis shuffled floor")
        plt.plot(xs, res["frame_shuf"], ":", color="gray", lw=1, label="frame shuffled floor")
        plt.xlabel("normalized depth (layer / final)"); plt.ylabel("probe balanced accuracy")
        plt.title(f"Causal vs nuisance decodability across depth\n{args.model} · {args.family}")
        plt.ylim(0, 1.02); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "step0_selectivity.png"), dpi=130)
        print(f"[saved] {outdir}/step0_selectivity.png + step0_results.json")
    except Exception as e:
        print(f"[plot skipped] {e}")

    print("\n===== GO / NO-GO =====")
    beh = "engages" if (cells / n) > max(copy_cells, zero_cells) / n + 0.02 else "WEAK"
    rep = "decodable" if best_axis > 0.65 else "WEAK"
    print(f"behavioral: {beh}  |  causal-latent decodable: {rep} (best axis probe {best_axis:.3f})")
    print("GO" if (rep == "decodable") else "REASSESS (see PLAN.md fallbacks)")


if __name__ == "__main__":
    main()
