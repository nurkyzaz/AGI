"""Orchestrate the Phase-1 pilot: 10 families x N seeds x 4 baselines.

Writes one JSONL line per (family, kind, seed) run to runs_causalarc/pilot/, and
an immutable summary RunRecord. Designed to shard trivially across a SLURM array
(pass --families / --seeds subsets), then aggregate.

    python -m causalarc.pilot.run_pilot --seeds 0-9 --steps 3000
    python -m causalarc.pilot.run_pilot --families F1_translate,F2_recolor_parity --seeds 0-1 --steps 500  # smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from ..families import FAMILIES
from ..records import RunRecord, write_record
from .train import PilotConfig, train_one

KINDS = ["oracle", "raw", "object_centric", "cond_ib"]
PILOT_DIR = Path(__file__).resolve().parent.parent.parent / "runs_causalarc" / "pilot"


def _parse_seeds(spec: str) -> List[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def run(families: List[str], seeds: List[int], kinds: List[str], cfg: PilotConfig,
        tag: str = "") -> List[dict]:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    shard = tag or f"{stamp}"
    per_run_path = PILOT_DIR / f"{shard}_runs.jsonl"

    rows = []
    total = len(families) * len(seeds) * len(kinds)
    i = 0
    for fname in families:
        fam = FAMILIES[fname]
        for seed in seeds:
            for kind in kinds:
                i += 1
                t0 = time.time()
                r = train_one(kind, fam, seed, cfg)
                r["wall_s"] = round(time.time() - t0, 1)
                rows.append(r)
                with per_run_path.open("a") as f:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
                print(f"[{i:>4}/{total}] {fname:20s} {kind:14s} seed={seed} "
                      f"iid={r['iid_exact']:.2f} shift={r['shifted_exact']:.2f} "
                      f"gap={r['ood_gap']:+.2f} Cprobe={r['causal_probe']:.2f} "
                      f"({r['wall_s']}s)", flush=True)

    record = RunRecord(
        kind="pilot_phase1",
        config={"families": families, "seeds": seeds, "kinds": kinds,
                "pilot_config": cfg.__dict__, "tag": shard},
        result={"n_runs": len(rows), "per_run_file": str(per_run_path)},
        seeds={"seeds": seeds},
    )
    write_record(record)
    print(f"\nwrote {len(rows)} runs -> {per_run_path}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="ALL")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    families = list(FAMILIES.keys()) if args.families == "ALL" else args.families.split(",")
    seeds = _parse_seeds(args.seeds)
    kinds = args.kinds.split(",")
    cfg = PilotConfig(steps=args.steps, batch=args.batch, device=args.device)
    run(families, seeds, kinds, cfg, tag=args.tag)


if __name__ == "__main__":
    main()
