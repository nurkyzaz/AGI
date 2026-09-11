"""Text in-context shortcut task (the plan's text-token analog).

Each demo pairs a SHAPE word and a COLOR word with a label. Within a task the
SHAPE->label map is the TRUE rule; the COLOR is a spurious cue perfectly aligned
with the label in the demos. The query is a CONFLICT: its shape implies one label,
its color another. Which label the model prefers reveals whether it generalized
via the rule (shape) or the shortcut (color).

Nonsense words + digit labels force genuine in-context learning (no lexical prior
about which cue "should" matter, and single-token labels make logit readout clean).
Self-contained, CPU-friendly (short prompts).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# pronounceable nonsense tokens (no strong label prior); labels are single tokens
SHAPES = ["wex", "jom", "fip", "vab", "kir", "dovern", "plim", "greb", "tannu", "yolp"]
COLORS = ["quv", "zol", "bik", "nel", "tor", "suan", "mig", "rax", "poth", "lend"]
LABELS = ["1", "2", "3", "4"]


@dataclass
class Task:
    prompt: str
    rule_label: str        # label implied by the query's SHAPE (the true rule)
    shortcut_label: str    # label implied by the query's COLOR (the spurious cue)
    conflict: bool
    shape_class: int
    color_class: int
    meta: dict


def make_task(rng: np.random.Generator, n_classes: int = 3, k_per: int = 2,
              conflict: bool = True) -> Task:
    shapes = list(rng.choice(SHAPES, size=n_classes, replace=False))
    colors = list(rng.choice(COLORS, size=n_classes, replace=False))
    labels = LABELS[:n_classes]
    # within-task the rule (shape->label) is a random bijection; color is aligned
    perm = rng.permutation(n_classes)
    shape_label = {shapes[i]: labels[perm[i]] for i in range(n_classes)}
    color_label = {colors[i]: labels[perm[i]] for i in range(n_classes)}  # aligned in demos
    cls_shape = {int(perm[i]): shapes[i] for i in range(n_classes)}
    cls_color = {int(perm[i]): colors[i] for i in range(n_classes)}

    demos = []
    for c in range(n_classes):
        for _ in range(k_per):
            demos.append((cls_shape[c], cls_color[c], labels[c]))
    rng.shuffle(demos)

    a = int(rng.integers(n_classes))                      # query shape class (rule)
    b = a if not conflict else int((a + 1 + rng.integers(n_classes - 1)) % n_classes)
    q_shape, q_color = cls_shape[a], cls_color[b]

    lines = [f"{s} {c} -> {l}" for (s, c, l) in demos]
    prompt = "\n".join(lines) + f"\n{q_shape} {q_color} ->"
    return Task(
        prompt=prompt, rule_label=labels[a], shortcut_label=labels[b],
        conflict=conflict, shape_class=a, color_class=b,
        meta={"q_shape": q_shape, "q_color": q_color,
              "shape_label": shape_label, "color_label": color_label},
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for conflict in (False, True):
        t = make_task(rng, conflict=conflict)
        print("=" * 60, "CONFLICT" if conflict else "ALIGNED")
        print(t.prompt)
        print(f"  rule_label={t.rule_label}  shortcut_label={t.shortcut_label}")
