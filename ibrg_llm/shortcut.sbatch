#!/bin/bash
# CPU-only array (no --gres=gpu) -> schedules immediately despite GPU contention.
# sbatch -> owned by the SLURM controller -> runs overnight regardless of laptop.
# One array task per seed; each runs the full H1/H2/H4 shortcut experiment.
# Submit from ~/agi:  sbatch ibrg_llm/shortcut.sbatch
#SBATCH --job-name=ibrg_sc
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/user/nurkyz/agi/ibrg_llm/logs/sc_%A_%a.out
#SBATCH --open-mode=truncate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/home/user/nurkyz/agi
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/user/nurkyz/agi
S=$SLURM_ARRAY_TASK_ID
echo "[sc] host=$(hostname) seed=$S start=$(date)"
/home/user/nurkyz/miniconda3/envs/agi/bin/python -m ibrg_llm.runner --device cpu --seed "$S" --tag "seed$S"
echo "[sc] seed=$S done=$(date)"
