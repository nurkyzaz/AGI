# Compression keeps the shortcut, not the rule: what tells them apart in a network's representation

Nurkyz Ydyrysova — MATS 12.0, Neel Nanda stream. Code, per-seed results, figures: https://github.com/nurkyzaz/AGI/tree/main/ibrg_llm

Terms, once: a **shortcut** predicts the label in the training examples but is not its cause, so it fails on new data; the **rule** is the cause and keeps working. **Compression (β)** limits how much a network keeps about its input. A **probe** is a linear classifier that reads one feature off frozen activations. **Intervention (do(U))** changes the training data so the shortcut no longer lines up with the label.

## The research question

When a network can solve a task two ways — by the rule or by a shortcut that only holds in the training data — is there anything inside the network that tells the two apart, and does that internal signal predict whether the network still works on new data where the shortcut is broken?

## Why it matters

Two questions on your list are special cases of this. "Why does a model pick one solution when several fit the training data" is shortcut versus rule. "Can a cheap probe tell us what the model is doing" is whether a probe is a usable monitor. I tried to answer both with baselines and with honest nulls.

## How I measured it

Two tracks. **Track A — a real language model:** Qwen2.5-Instruct at 0.5B, 1.5B, 3B. The task is in-context text classification with k labeled examples per class; each item carries a rule feature (the true label) and a shortcut feature (a cue that matches the label in the examples). The "agree" form (rule and shortcut match) checks the task is doable; the "conflict" form (they disagree) measures which one the model follows. I train linear probes on frozen hidden states and project feature directions out of the activations to test cause. **Track B — a network I train from scratch**, where I know the true rule: encoder → limited channel z → classifier, trained with cross-entropy + β·(size of z), so β sets how much the network may keep. The input is [rule block | extra block] and the label depends only on the rule block; the extra block is set to useless noise, a cheap shortcut, or the same shortcut with the intervention applied. The new-data test always breaks the shortcut.

## Experiments and what they showed

### Track A — the language model

**Experiment 1 — Does the model follow the rule or the shortcut, and does that change with size?** I measured shortcut-use under conflict (fraction of conflict items where it follows the shortcut; 0.5 = follows rule and shortcut equally) across the three sizes.

[[FIG:fig_scale.png]]
Figure 1. Shortcut-use under conflict falls as the model grows: 0.552, 0.501, 0.491 for 0.5B, 1.5B, 3B. At 0.5 the model has no net preference; the 3B is there. Larger models rely on the shortcut less.

**What it showed:** the 3B model follows the rule (shortcut-use 0.49) and can do the task (0.997 when rule and shortcut agree, so it is not simply failing), and reliance on the shortcut drops monotonically with size.

**Experiment 2 — Where is the shortcut inside the network, and does it cause the behavior?** I projected the shortcut directions out of each layer and compared the effect to removing a random direction of the same length (the baseline that makes the claim non-trivial).

[[FIG:fig_base_H1_H4.png]]
Figure 2. Right: the removal test — projecting the shortcut directions out of a middle layer changes behavior by +0.018, while projecting out a random direction of equal length changes it by +0.001 (about twenty times less), so those directions carry the shortcut and this is not a generic effect of perturbing the activations. Left (behavioral recap): the 3B does not prefer the shortcut under conflict (0.49) and scores 0.997 when rule and shortcut agree.

[[FIG:fig_ablsweep.png]]
Figure 3. Removing the shortcut directions changes behavior most at the middle layers (left), and the model's output is specifically sensitive to perturbing that subspace (right). The shortcut lives in a small, mid-network set of directions.

**What it showed:** removing the shortcut directions moves behavior about twenty times more than an equal-size random direction (+0.018 vs +0.001), concentrated in the middle layers — so the subspace is causal, not merely readable.

**Experiment 3 — When does the model lean on the shortcut?** I swept how strong the shortcut cue is and how many in-context examples the model sees.

[[FIG:fig_phase.png]]
Figure 4. Shortcut-use rises with shortcut strength (down the rows) and falls with more in-context examples (across the columns). The model uses the shortcut exactly when it is strong and the evidence is thin.

**What it showed:** shortcut-use is not fixed — it grows with cue strength and shrinks with more evidence, as a rational-shortcut account predicts.

**Experiment 4 — Does a probe gap predict generalization?** For each model and seed I measured (rule-probe accuracy − shortcut-probe accuracy) from frozen activations, and how often the model follows the rule.

[[FIG:fig_selectivity.png]]
Figure 5. Each point is one model at one seed. The more a linear probe reads the rule better than the shortcut (horizontal), the more the model follows the rule (vertical); they rise together at r = 0.68. Most of the spread is between the three sizes, so this partly restates "bigger models follow the rule more" — a limitation I flag.

**What it showed:** the probe gap tracks rule-following (r = 0.68), but mostly across sizes, so it is confounded with scale.

**Experiment 5 — Is a shortcut probe a usable monitor of shortcut use?** I compared how readable the shortcut cue is from the activations to how much the model actually uses it.

[[FIG:fig_monitor.png]]
Figure 6. If readability were a monitor, points would rise to the right. Instead the correlation is −0.34 — the wrong direction. How readable the shortcut is does not tell you how much the model uses it.

**What it showed (disproven):** the shortcut probe is not a monitor. A feature being linearly present in the activations is not evidence it drives behavior.

### Track B — the controlled network, where I know the true rule

**Experiment 6 — Does compression discard a useless feature and help generalization?** The extra input is pure noise. I raised β and watched what the representation keeps and how it does on new data.

[[FIG:fig_vib.png]]
Figure 7. As compression rises (left to right), the noise becomes unreadable from the representation (0.63 → 0.00) while the rule stays readable (~0.87). But new-data accuracy barely moves (best 0.73). Right: rate against new-data accuracy — spending fewer bits costs almost no accuracy, and buys almost none.

**What it showed (disproven that compression generically helps):** compression cleanly removes the noise and keeps the rule, yet generalization barely changes. Discarding a feature helps only if that feature was hurting you.

**Experiment 7 — Can compression fix a cheap shortcut? (strongest result)** The extra input is a cheap shortcut that predicts the training label perfectly and is broken on new data. I swept β; then I changed the training data so the shortcut no longer lines up with the label (the intervention, do(U)) and swept again.

[[FIG:fig_boundary.png]]
Figure 8 — the strongest result. Left: raising compression across four orders of magnitude leaves new-data accuracy at chance at every setting (blue); the shortcut stays fully readable throughout (right). The bottleneck does not drop the shortcut — it keeps it, because on the training data the rule and the shortcut are equally good, equally small descriptions. Orange: changing the training data so the shortcut no longer lines up with the label restores generalization, 0.46 → 0.90.

**What it showed:** compression alone cannot fix a cheap shortcut — it keeps it at every β. The intervention fixes it. Compression is not enough; intervention is required.

**Experiment 8 — Inside the representation, what tells the rule from the shortcut?** I built one direction that carries the rule and one that carries the shortcut, then scored each two ways: read the label off it on the training data (observation), and predict the label from it across environments that line the shortcut up with the label to different degrees (intervention).

[[FIG:fig_foliation.png]]
Figure 9. Right (shortcut-aligned training data): the rule direction and the shortcut direction read off the training data identically — both recover the label at 1.00, so a score trying to tell them apart sits at 0.48 (0.5 = cannot tell). Across environments they separate: the rule predicts the label everywhere (0.83), the shortcut only where it lines up (0.73). Left (a network trained with the intervention, which generalizes): the rule-vs-shortcut separation does not peak at the compression that generalizes best — an honest null.

[[FIG:fig_fdt.png]]
Figure 10. I expected a special "critical" compression where the network is most sensitive to a nudge. I do not find one: the response to a small nudge of the representation and the spread of the representation both fall smoothly as compression rises, with no peak.

**What it showed:** the rule is invisible to observation (separation 0.48) and visible to intervention (0.83 vs 0.73). The "criticality" idea I started from is not supported — no special compression point.

## Conclusion

Compression picks the smallest description that fits the training data, and a shortcut is such a description, so the bottleneck keeps it. What separates the rule from the shortcut is not anything you can read from one training set — it is how each behaves when you change the data. "Is the model using the rule" is therefore not an observation question in the hard case, and probe-based monitoring inherits that limit. Turning information into a law needs compression **and** intervention, not compression alone.

## Strongest evidence against what I expected

The clean hope — "compression drops the shortcut and rescues generalization" — is false in the decisive test (Experiment 7): the bottleneck keeps the cheap shortcut at every setting. This is the correct behavior of the compression objective, and it is why my conclusion moved from "compression" to "compression and intervention." The monitor null (Experiment 5) and the no-critical-point null (Experiment 8) are the other two.

## Biggest limitations (honest)

- Models are ≤3B and the task is one synthetic family, so the scale trend and effect sizes may not transfer. Addressable with a 7–14B model and more task families; the code scales unchanged — this was a disk-and-GPU-time limit, not a design choice.
- The probe-gap result (Experiment 4) is mostly across three sizes, so it is confounded with size. Addressable and planned: the same correlation within one model across seeds, holding task accuracy fixed, and a removal test that turns the correlation into cause. I did not finish this.
- The clean compression results (7, 8) are in a network I train from scratch, not a language model. That isolates the mechanism but leaves the language-model case motivated, not proven.
- Information Bottleneck = Renormalization Group is my reason for looking; I did not test the physics equivalence.

## Reproduce

All code, per-seed results, and figures: https://github.com/nurkyzaz/AGI/tree/main/ibrg_llm . Track A: `shortcut_task.py`, `runner.py`, `phase.py`, `ablsweep.py`. Track B: `vib.py` (noise), `boundary.py` (compression vs intervention), `crit.py` (rule-vs-shortcut directions). Regenerate every figure with `python -m ibrg_llm.aggregate`. Every number above is an average over the saved per-seed files.
