#!/bin/bash
# Robust batch submission for Step 0 (owned by the SLURM controller, so it runs
# whenever a GPU frees regardless of the laptop / any SSH session).
# Submit from ~/agi:  sbatch ibrg_llm/step0.sbatch
#SBATCH --job-name=ibrg_s0
#SBATCH --partition=a,b,c,normal
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/user/nurkyz/agi/ibrg_llm/logs/step0_sbatch.out
#SBATCH --open-mode=truncate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/user/nurkyz/agi
cd /home/user/nurkyz/agi
echo "[sbatch] host=$(hostname) start=$(date)"
/home/user/nurkyz/miniconda3/envs/agi/bin/python -m ibrg_llm.step0
echo "[sbatch] done=$(date)"
