"""GATE 0 driver — the pre-registered pass/fail decision for Phase 0.

Runs all three GATE 0 checks and writes one immutable run record:

  1. Every declared intervention passes its generator test
     (C variables can change the output; U variables never do).
  2. A planted shortcut is invisible to i.i.d. accuracy but fails under
     correlation reversal at test.
  3. The metric code recovers the known causal mask on planted families
     (precision/recall == 1 on ground truth we inserted).

ALL must hold to enter Phase 1. Run:  python -m causalarc.gate0
"""
from __future__ import annotations

import argparse
from typing import Any, Dict

from .families import FAMILIES
from .interventions import intervention_tests, recover_causal_mask
from .records import RunRecord, write_record
from .splits import accuracy, causal_predict, make_shortcut_splits, shortcut_predict

# thresholds for the shortcut check (pre-registered)
IID_HIGH = 0.95   # shortcut must succeed on train/iid
SHIFT_LOW = 0.10  # shortcut must collapse under reversal


def check_interventions(n_content: int, n_base: int, seed: int) -> Dict[str, Any]:
    detail = {}
    passed = True
    for name, fam in FAMILIES.items():
        ok, per_var = intervention_tests(fam, n_content=n_content, n_base=n_base, seed=seed)
        detail[name] = {"passed": ok, "variables": per_var}
        passed = passed and ok
    return {"passed": passed, "families": detail}


def check_shortcut(n_per_split: int, seed: int) -> Dict[str, Any]:
    splits = make_shortcut_splits(n_per_split=n_per_split, seed=seed)
    sc = {k: accuracy(shortcut_predict, v) for k, v in splits.items()}
    ca = {k: accuracy(causal_predict, v) for k, v in splits.items()}
    passed = (
        sc["train"] >= IID_HIGH
        and sc["iid"] >= IID_HIGH
        and sc["shifted"] <= SHIFT_LOW
        and ca["shifted"] >= IID_HIGH  # sanity: the causal signal survives the shift
    )
    return {
        "passed": passed,
        "shortcut_accuracy": sc,
        "causal_accuracy": ca,
        "thresholds": {"iid_high": IID_HIGH, "shift_low": SHIFT_LOW},
    }


def check_mask(n_content: int, n_base: int, seed: int) -> Dict[str, Any]:
    detail = {}
    passed = True
    for name, fam in FAMILIES.items():
        res = recover_causal_mask(fam, n_content=n_content, n_base=n_base, seed=seed)
        detail[name] = {
            "precision": res.precision,
            "recall": res.recall,
            "effect_rates": res.effect_rates,
            "predicted_causal": res.predicted_causal,
            "ground_truth_causal": res.ground_truth_causal,
        }
        passed = passed and res.perfect
    return {"passed": passed, "families": detail}


def run_gate0(n_content: int = 40, n_base: int = 3, n_split: int = 120, seed: int = 0,
              write: bool = True) -> Dict[str, Any]:
    item1 = check_interventions(n_content, n_base, seed)
    item2 = check_shortcut(n_split, seed)
    item3 = check_mask(n_content, n_base, seed)
    overall = item1["passed"] and item2["passed"] and item3["passed"]

    result = {
        "overall_pass": overall,
        "item1_interventions": item1,
        "item2_shortcut": item2,
        "item3_causal_mask": item3,
    }
    config = {"n_content": n_content, "n_base": n_base, "n_split": n_split,
              "families": list(FAMILIES.keys())}
    ledger_hashes = {name: fam.ledger().hash() for name, fam in FAMILIES.items()}
    record = RunRecord(kind="gate0", config=config,
                       result=result, seeds={"seed": seed, "ledger_hashes": ledger_hashes})
    if write:
        path = write_record(record)
        result["_record_path"] = str(path)
        result["_record_hash"] = record.content_hash()
    return result


def _print_report(r: Dict[str, Any]) -> None:
    mark = lambda b: "PASS" if b else "FAIL"
    print("=" * 68)
    print(f"GATE 0  —  overall: {mark(r['overall_pass'])}")
    print("=" * 68)

    print(f"\n[item 1] intervention tests ............... {mark(r['item1_interventions']['passed'])}")
    for name, d in r["item1_interventions"]["families"].items():
        print(f"  {name:22s} {mark(d['passed'])}")
        for v, info in d["variables"].items():
            print(f"      {v:18s} role={info['role']} rate={info['effect_rate']:.3f} "
                  f"{'ok' if info['ok'] else 'BAD'}")

    s = r["item2_shortcut"]
    print(f"\n[item 2] planted shortcut ................. {mark(s['passed'])}")
    print(f"  shortcut acc: {s['shortcut_accuracy']}")
    print(f"  causal   acc: {s['causal_accuracy']}")

    m = r["item3_causal_mask"]
    print(f"\n[item 3] causal-mask recovery ............. {mark(m['passed'])}")
    for name, d in m["families"].items():
        print(f"  {name:22s} P={d['precision']:.2f} R={d['recall']:.2f} "
              f"pred_causal={d['predicted_causal']}")

    if "_record_path" in r:
        print(f"\nrecord: {r['_record_path']}  ({r['_record_hash']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run GATE 0 for causalarc Phase 0.")
    ap.add_argument("--n-content", type=int, default=40)
    ap.add_argument("--n-base", type=int, default=3)
    ap.add_argument("--n-split", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    r = run_gate0(args.n_content, args.n_base, args.n_split, args.seed, write=not args.no_write)
    _print_report(r)
    raise SystemExit(0 if r["overall_pass"] else 1)


if __name__ == "__main__":
    main()
