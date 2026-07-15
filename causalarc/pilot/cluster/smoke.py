"""Repair-gate smoke (B-1.1 v2): can the deeper decoder fit the two families the
v1 oracle failed on (F4_gravity 0.03, F6_rotate 0.18)? Full v2 budget, oracle
only. Gate: oracle iid_exact >= 0.50 on both, else iterate the repair before
burning the full wave. Run under srun on a GPU node.
"""
import time

import torch

from causalarc.families import FAMILIES
from causalarc.pilot.train import PilotConfig, train_one

print("cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
cfg = PilotConfig(device="cuda")  # v2 defaults: steps=4000, hidden=64, 2 attn blocks
for f in ["F4_gravity", "F6_rotate", "F1_translate"]:  # F1 = regression control
    t = time.time()
    r = train_one("oracle", FAMILIES[f], 0, cfg)
    print(f"{f:20s} oracle iid_exact={r['iid_exact']:.2f} cell={r['iid_cell']:.3f} "
          f"({time.time()-t:.0f}s)", flush=True)
