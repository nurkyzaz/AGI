#!/bin/bash
# Overnight cross-family probe sweep as a SLURM ARRAY job: one submission, each
# array task runs one geometric family (clean color-canonicalization) and writes
# its own selectivity plot + numbers. Runs whenever GPUs free, laptop-independent.
# Submit from ~/agi:  sbatch ibrg_llm/families.sbatch
#SBATCH --job-name=ibrg_fam
#SBATCH --partition=a,b,c,normal
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=/home/user/nurkyz/agi/ibrg_llm/logs/fam_%A_%a.out
#SBATCH --open-mode=truncate
FAMS=(F1_translate F3_reflect F4_cycle F8_diag)
F=${FAMS[$SLURM_ARRAY_TASK_ID]}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/home/user/nurkyz/agi
cd /home/user/nurkyz/agi
echo "[fam] host=$(hostname) family=$F start=$(date)"
/home/user/nurkyz/miniconda3/envs/agi/bin/python -m ibrg_llm.step0 --family "$F" --tag "$F"
echo "[fam] family=$F done=$(date)"
