#!/bin/bash
#SBATCH --job-name=ibrg_q05b
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/user/nurkyz/agi/ibrg_llm/logs/q05b_%A_%a.out
#SBATCH --open-mode=truncate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/home/user/nurkyz/agi OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/user/nurkyz/agi
S=$SLURM_ARRAY_TASK_ID
/home/user/nurkyz/miniconda3/envs/agi/bin/python -m ibrg_llm.runner --device cpu --model Qwen/Qwen2.5-0.5B-Instruct --seed $S --tag q05b_s$S
