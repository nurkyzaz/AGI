#!/bin/bash
#SBATCH --job-name=ibrg_vib
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/user/nurkyz/agi/ibrg_llm/logs/vib_%A_%a.out
#SBATCH --open-mode=truncate
export PYTHONPATH=/home/user/nurkyz/agi OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/user/nurkyz/agi
S=$SLURM_ARRAY_TASK_ID
echo "[vib] host=$(hostname) seed=$S start=$(date)"
/home/user/nurkyz/miniconda3/envs/agi/bin/python -m ibrg_llm.vib --seed "$S" --tag "seed$S"
echo "[vib] seed=$S done=$(date)"
