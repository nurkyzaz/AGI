# === IB=RG probe sweep — self-contained (free Colab T4 GPU) ============
# Does a real LLM selectively preserve the CAUSAL rule of a few-shot task while
# discarding a SURFACE NUISANCE, across depth? Probe both from the residual
# stream. Self-contained: no repo needed. Paste into a Colab GPU cell and run.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.44", "accelerate"], check=False)

import numpy as np, torch, re, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

MODEL = "Qwen/Qwen2.5-3B-Instruct"     # fits free T4 (fp16 ~6GB)
GRID, OBJ, K = 12, 7, 3
N_PROBE, N_BEHAV = 240, 30
assert torch.cuda.is_available(), "Runtime > Change runtime type > T4 GPU"

# ---- self-contained stimulus generator -------------------------------------
# Causal rule = WHICH geometric transform maps input->output (must be inferred
# from the demos). Nuisance = frame_color (surface, printed in every grid).
TRANSFORMS = {"flip_h": lambda g: np.fliplr(g), "flip_v": lambda g: np.flipud(g),
              "rot180": lambda g: np.rot90(g, 2), "transpose": lambda g: g.T}
TNAMES = list(TRANSFORMS)

def _shape(rng):
    g = np.zeros((GRID, GRID), int)
    r, c = rng.integers(3, 9), rng.integers(3, 9)
    cells = {(int(r), int(c))}
    target = rng.integers(4, 8)
    for _ in range(target * 5):
        if len(cells) >= target:
            break
        dr, dc = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        nr, nc = int(r + dr), int(c + dc)
        if 2 <= nr <= 9 and 2 <= nc <= 9:
            cells.add((nr, nc)); r, c = nr, nc
    for rr, cc in cells:
        g[rr, cc] = OBJ
    return g

def _asym(rng):
    for _ in range(80):
        g = _shape(rng)
        if not any(np.array_equal(g, T(g)) for T in TRANSFORMS.values()):
            return g
    return g

def _g2t(g):
    return "\n".join(" ".join(map(str, row)) for row in g)

def make_stim(rng, k=K):
    tname = TNAMES[rng.integers(len(TNAMES))]; T = TRANSFORMS[tname]
    pool = [c for c in range(1, 10) if c != OBJ]; rng.shuffle(pool)
    frame_c, dist_c = int(pool[0]), int(pool[1])
    def one():
        s = _asym(rng)
        out = np.array(T(s))
        inp = s.copy()
        dr, dc = int(rng.integers(1, 11)), int(rng.integers(1, 11))
        if inp[dr, dc] == 0:
            inp[dr, dc] = dist_c                                  # distractor
        inp[0, :] = inp[-1, :] = inp[:, 0] = inp[:, -1] = frame_c  # frame ring
        return inp, out
    demos = [one() for _ in range(k)]
    qi, qo = one()
    P = ("You are given input/output grid pairs that all follow one hidden rule.\n"
         "Grids are rows of digits 0-9 (0=background). Infer the rule from the\n"
         "examples, then produce the output grid for the final input.\n")
    for i, (gi, go) in enumerate(demos):
        P += f"\nExample {i+1}\nInput:\n{_g2t(gi)}\nOutput:\n{_g2t(go)}"
    P += f"\nNow the final input.\nInput:\n{_g2t(qi)}\nOutput:\n"
    return P, _g2t(qo), tname, frame_c, qi

def parse_grid(text):
    rows = []
    for line in text.splitlines():
        nums = re.findall(r"-?\d+", line)
        if len(nums) >= GRID:
            rows.append([int(x) for x in nums[:GRID]])
        if len(rows) == GRID:
            break
    return np.array(rows) if len(rows) == GRID else None

# ---- load model ------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
print("loading", MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda", output_hidden_states=True).eval()
dev = next(model.parameters()).device
def fmt(p):
    return tok.apply_chat_template([{"role": "user", "content": p}],
                                   tokenize=False, add_generation_prompt=True)

# ---- (A) behavioral --------------------------------------------------------
rng = np.random.default_rng(0)
ex = cells = copy = zero = wf = 0
for _ in range(N_BEHAV):
    P, tgt_t, _, _, qi = make_stim(rng)
    ids = tok(fmt(P), return_tensors="pt").to(dev)
    with torch.no_grad():
        o = model.generate(**ids, max_new_tokens=200, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    pred = parse_grid(tok.decode(o[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
    tgt = np.array([[int(x) for x in r.split()] for r in tgt_t.splitlines()])
    wf += pred is not None
    ex += int(pred is not None and np.array_equal(pred, tgt))
    cells += 0.0 if pred is None else float((pred == tgt).mean())
    copy += float((qi == tgt).mean()); zero += float((tgt == 0).mean())
print(f"\n(A) BEHAVIORAL  exact={ex/N_BEHAV:.3f} cell={cells/N_BEHAV:.3f} "
      f"copy_baseline={copy/N_BEHAV:.3f} zero_baseline={zero/N_BEHAV:.3f} wellformed={wf/N_BEHAV:.3f}")

# ---- (B) probe causal (transform) vs nuisance (frame) across depth ---------
rng = np.random.default_rng(1)
H, yc, yu = [], [], []
for i in range(N_PROBE):
    P, _, tname, frame_c, _ = make_stim(rng)
    ids = tok(fmt(P), return_tensors="pt").to(dev)
    with torch.no_grad():
        hs = model(**ids).hidden_states
    H.append(torch.stack([h[0, -1, :] for h in hs]).float().cpu().numpy())
    yc.append(tname); yu.append(frame_c)
    if (i + 1) % 40 == 0:
        print(f"  probe {i+1}/{N_PROBE}")
H = np.stack(H); yc = LabelEncoder().fit_transform(yc); yu = np.array(yu)
cch, uch = 1/len(set(yc)), 1/len(set(yu))

def probe(X, y):
    ns = int(min(5, np.bincount(y)[np.bincount(y) > 0].min()))
    if ns < 2:
        return float("nan")
    a = [balanced_accuracy_score(y[te], LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict(X[te]))
         for tr, te in StratifiedKFold(ns, shuffle=True, random_state=0).split(X, y)]
    return float(np.mean(a))

L = H.shape[1]
causal = [probe(H[:, i, :], yc) for i in range(L)]
nuis = [probe(H[:, i, :], yu) for i in range(L)]
c_sh = [probe(H[:, i, :], np.random.default_rng(0).permutation(yc)) for i in range(L)]
u_sh = [probe(H[:, i, :], np.random.default_rng(0).permutation(yu)) for i in range(L)]
print(f"\n(B) PROBES  causal=transform(chance {cch:.2f}) nuisance=frame(chance {uch:.2f})")
print("best causal =", round(np.nanmax(causal), 3), "| best nuisance =", round(np.nanmax(nuis), 3))

xs = [i/(L-1) for i in range(L)]
plt.figure(figsize=(7, 4.5))
plt.plot(xs, causal, "-o", ms=3, label="causal (transform rule, inferred)")
plt.plot(xs, nuis, "-s", ms=3, label="nuisance (frame color, surface)")
plt.plot(xs, c_sh, "--", color="gray", lw=1, label="shuffled floors")
plt.plot(xs, u_sh, ":", color="gray", lw=1)
plt.axhline(cch, color="C0", ls=":", lw=0.8); plt.axhline(uch, color="C1", ls=":", lw=0.8)
plt.xlabel("normalized depth"); plt.ylabel("probe balanced accuracy")
plt.title(f"Causal vs nuisance decodability across depth\n{MODEL}"); plt.ylim(0, 1.02)
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("selectivity.png", dpi=130); plt.show()
json.dump({"behavioral": {"exact": ex/N_BEHAV, "cell": cells/N_BEHAV, "wellformed": wf/N_BEHAV},
           "causal": causal, "nuis": nuis, "causal_shuf": c_sh, "nuis_shuf": u_sh,
           "chance": {"causal": cch, "nuis": uch}}, open("results.json", "w"), indent=2)
print("\nSaved selectivity.png + results.json  (download and send to Claude)")
