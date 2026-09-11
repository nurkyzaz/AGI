#!/bin/bash
#SBATCH --job-name=ibrg_abl
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/user/nurkyz/agi/ibrg_llm/logs/abl_%A_%a.out
#SBATCH --open-mode=truncate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/home/user/nurkyz/agi OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/user/nurkyz/agi
S=$SLURM_ARRAY_TASK_ID
echo "[abl] host=$(hostname) seed=$S start=$(date)"
/home/user/nurkyz/miniconda3/envs/agi/bin/python -m ibrg_llm.ablsweep --seed "$S" --tag "seed$S"
echo "[abl] seed=$S done=$(date)"
