"""Quick GPU learnability check: oracle on representative families.

Confirms the attention decoder can learn recolor, translate, reflect, and rotate
before we commit to the full sweep. Run under srun on a GPU node.
"""
import time

import torch

from causalarc.families import FAMILIES
from causalarc.pilot.train import PilotConfig, train_one

print("cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
cfg = PilotConfig(steps=1200, batch=64, eval_n=128, probe_n=256, probe_steps=150, device="cuda")
for f in ["F2_recolor_parity", "F1_translate", "F3_reflect", "F6_rotate"]:
    t = time.time()
    r = train_one("oracle", FAMILIES[f], 0, cfg)
    print(f"{f:20s} oracle iid_exact={r['iid_exact']:.2f} cell={r['iid_cell']:.3f} "
          f"({time.time()-t:.0f}s)", flush=True)
