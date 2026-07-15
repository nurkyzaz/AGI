#!/bin/bash
#SBATCH --job-name=agi_pilot
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/user/nurkyz/agi/slurm_logs/pilot_%A_%a.out
#SBATCH --error=/home/user/nurkyz/agi/slurm_logs/pilot_%A_%a.out

# Phase-1 pilot. QOS caps the user at 8 jobs, so we pack 2 families per array
# task (5 tasks): each trains 2 families x 10 seeds x 4 baselines on one GPU.
# Aggregate afterward with gate1.py.
set -euo pipefail

GROUPS=("F1_translate,F2_recolor_parity" \
        "F3_reflect,F4_gravity" \
        "F5_shortcut,F6_rotate" \
        "F7_recolor_size,F8_diag" \
        "F9_colormap,F10_recolor_half")
PAIR=${GROUPS[$SLURM_ARRAY_TASK_ID]}

RUN=/home/user/nurkyz/agi/causalarc/pilot/cluster/run.sh
echo "host=$(hostname) families=$PAIR start=$(date)"
bash "$RUN" -m causalarc.pilot.run_pilot \
    --families "$PAIR" --seeds 0-9 --steps 3000 --device cuda \
    --tag "pilot_g${SLURM_ARRAY_TASK_ID}"
echo "done=$(date)"
