#!/bin/bash
#SBATCH --job-name=ibrg_bnd
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/user/nurkyz/agi/ibrg_llm/logs/bnd_%A_%a.out
#SBATCH --open-mode=truncate
export PYTHONPATH=/home/user/nurkyz/agi OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/user/nurkyz/agi
S=$SLURM_ARRAY_TASK_ID
/home/user/nurkyz/miniconda3/envs/agi/bin/python -m ibrg_llm.boundary --seed "$S" --tag "seed$S"
