"""GATE 1 evaluation — does a causal-compression signal exist at all?

Aggregates the per-run pilot JSONL and evaluates the three pre-registered GATE 1
items with the TASK FAMILY as the independent unit (never the generated grids):

  1. Interventions produce a nontrivial OOD gap: the raw baseline drops
     meaningfully from iid to shifted (do(U)+recombination).
  2. The oracle beats the raw baseline under shift (knowing C helps OOD).
  3. At least one causal-score metric is stable across seeds (not seed-noise).

Thresholds are frozen in PREREGISTRATION.md (P1.4) and imported here so the gate
cannot be silently moved after seeing results.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..records import RunRecord, write_record

PILOT_DIR = Path(__file__).resolve().parent.parent.parent / "runs_causalarc" / "pilot"

# ---- pre-registered thresholds (see PREREGISTRATION.md) ---------------------
T_RAW_GAP = 0.15          # pooled raw (iid-shifted) exact-match drop
T_RAW_GAP_CI_LOW = 0.05   # 95% family-cluster bootstrap lower bound must exceed
T_ORACLE_MARGIN = 0.15    # pooled (oracle_shifted - raw_shifted)
T_PROBE_STD = 0.10        # max mean cross-seed std of causal_probe (stability)
T_PROBE_MEAN = 0.50       # min mean causal_probe (well above chance)
FRAC_FAMILIES_DROP = 0.7  # fraction of families where raw shifted < raw iid


def load_runs(files: List[str]) -> List[dict]:
    rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _by_family_kind(rows: List[dict]) -> Dict:
    """(family, kind) -> {metric: [values over seeds]}."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["family"], r["kind"])
        for m in ("iid_exact", "shifted_exact", "ood_gap", "causal_probe", "nuisance_probe"):
            agg[key][m].append(r[m])
    return agg


def _family_means(agg, kind: str, metric: str) -> Dict[str, float]:
    out = {}
    for (fam, k), metrics in agg.items():
        if k == kind:
            out[fam] = float(np.mean(metrics[metric]))
    return out


def _cluster_bootstrap(per_family: Dict[str, float], n_boot: int = 10000, seed: int = 0):
    """Percentile 95% CI over families (the independent unit)."""
    rng = np.random.default_rng(seed)
    vals = np.array(list(per_family.values()))
    fams = len(vals)
    means = np.array([rng.choice(vals, size=fams, replace=True).mean() for _ in range(n_boot)])
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_gate1(rows: List[dict]) -> Dict:
    agg = _by_family_kind(rows)

    # item 1: raw OOD gap
    raw_gap = _family_means(agg, "raw", "ood_gap")
    raw_iid = _family_means(agg, "raw", "iid_exact")
    raw_shift = _family_means(agg, "raw", "shifted_exact")
    gap_mean, gap_lo, gap_hi = _cluster_bootstrap(raw_gap)
    frac_drop = np.mean([raw_shift[f] < raw_iid[f] for f in raw_iid])
    item1 = {
        "raw_ood_gap_mean": gap_mean, "ci95": [gap_lo, gap_hi],
        "frac_families_drop": float(frac_drop),
        "per_family_gap": raw_gap,
        "passed": bool(gap_mean >= T_RAW_GAP and gap_lo > T_RAW_GAP_CI_LOW
                       and frac_drop >= FRAC_FAMILIES_DROP),
    }

    # item 2: oracle beats raw under shift
    oracle_shift = _family_means(agg, "oracle", "shifted_exact")
    margin = {f: oracle_shift[f] - raw_shift[f] for f in oracle_shift}
    m_mean, m_lo, m_hi = _cluster_bootstrap(margin)
    item2 = {
        "oracle_minus_raw_shifted_mean": m_mean, "ci95": [m_lo, m_hi],
        "oracle_shifted": oracle_shift, "raw_shifted": raw_shift,
        "passed": bool(m_mean >= T_ORACLE_MARGIN and m_lo > 0.0),
    }

    # item 3: causal-score stability across seeds
    probe_std, probe_mean = {}, {}
    for (fam, k), metrics in agg.items():
        if k == "raw":
            probe_std[fam] = float(np.std(metrics["causal_probe"]))
            probe_mean[fam] = float(np.mean(metrics["causal_probe"]))
    mean_std = float(np.mean(list(probe_std.values()))) if probe_std else 1.0
    mean_probe = float(np.mean(list(probe_mean.values()))) if probe_mean else 0.0
    item3 = {
        "mean_cross_seed_std": mean_std, "mean_causal_probe": mean_probe,
        "per_family_std": probe_std,
        "passed": bool(mean_std <= T_PROBE_STD and mean_probe >= T_PROBE_MEAN),
    }

    # descriptive: object-centric robustness (supports the invariance story)
    oc_shift = _family_means(agg, "object_centric", "shifted_exact")
    descriptive = {
        "object_centric_shifted_mean": float(np.mean(list(oc_shift.values()))) if oc_shift else None,
        "raw_shifted_mean": float(np.mean(list(raw_shift.values()))) if raw_shift else None,
    }

    overall = item1["passed"] and item2["passed"] and item3["passed"]
    return {"overall_pass": overall, "item1_ood_gap": item1,
            "item2_oracle_beats_raw": item2, "item3_causal_score_stable": item3,
            "descriptive": descriptive, "n_runs": len(rows)}


def _print(res: Dict) -> None:
    mk = lambda b: "PASS" if b else "FAIL"
    print("=" * 68)
    print(f"GATE 1 — overall: {mk(res['overall_pass'])}   (n_runs={res['n_runs']})")
    print("=" * 68)
    i1 = res["item1_ood_gap"]
    print(f"\n[item 1] raw OOD gap ............... {mk(i1['passed'])}")
    print(f"  raw iid-shifted gap = {i1['raw_ood_gap_mean']:.3f}  95%CI {i1['ci95']}")
    print(f"  families that drop  = {i1['frac_families_drop']*100:.0f}%")
    i2 = res["item2_oracle_beats_raw"]
    print(f"\n[item 2] oracle beats raw (shift) . {mk(i2['passed'])}")
    print(f"  oracle-raw shifted margin = {i2['oracle_minus_raw_shifted_mean']:.3f}  95%CI {i2['ci95']}")
    i3 = res["item3_causal_score_stable"]
    print(f"\n[item 3] causal score stable ...... {mk(i3['passed'])}")
    print(f"  mean cross-seed std = {i3['mean_cross_seed_std']:.3f}  mean probe = {i3['mean_causal_probe']:.3f}")
    d = res["descriptive"]
    print(f"\n[descriptive] object-centric shifted={d['object_centric_shifted_mean']} "
          f"vs raw shifted={d['raw_shifted_mean']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(PILOT_DIR / "*_runs.jsonl"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no run files match {args.glob}")
    rows = load_runs(files)
    res = evaluate_gate1(rows)
    _print(res)
    if not args.no_write:
        rec = RunRecord(kind="gate1", config={"files": files},
                        result=res, seeds={})
        p = write_record(rec)
        print(f"\nrecord: {p}")
    raise SystemExit(0 if res["overall_pass"] else 1)


if __name__ == "__main__":
    main()
