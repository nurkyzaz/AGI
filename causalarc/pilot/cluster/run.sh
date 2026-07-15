#!/usr/bin/env bash
# Run a python script/module in the `agi` conda env from the repo root.
# Self-contained: no conda activation (login shell is tcsh), uses the env python
# directly and puts the repo root on PYTHONPATH so `import causalarc` resolves.
# Usage: bash run.sh <script.py|-m module> [args...]
set -euo pipefail
PY=/home/user/nurkyz/miniconda3/envs/agi/bin/python
export PYTHONPATH="$HOME/agi:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$HOME/agi"
exec "$PY" "$@"
