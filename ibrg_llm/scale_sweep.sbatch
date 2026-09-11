#!/bin/bash
# Wave 1 / A1: model-scale sweep on CPU. 3B is already covered by the base array,
# so this adds {0.5B, 1.5B} x seeds 0-4. Array index -> (model, seed).
# CPU-only -> schedules instantly; sbatch -> laptop-independent.
#SBATCH --job-name=ibrg_scale
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/user/nurkyz/agi/ibrg_llm/logs/scale_%A_%a.out
#SBATCH --open-mode=truncate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/home/user/nurkyz/agi
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/user/nurkyz/agi
MODELS=(Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct)
ABBR=(q05b q15b)
mi=$(( SLURM_ARRAY_TASK_ID / 5 )); si=$(( SLURM_ARRAY_TASK_ID % 5 ))
M=${MODELS[$mi]}; A=${ABBR[$mi]}
echo "[scale] host=$(hostname) model=$M seed=$si start=$(date)"
/home/user/nurkyz/miniconda3/envs/agi/bin/python -m ibrg_llm.runner --device cpu --model "$M" --seed "$si" --tag "${A}_s${si}"
echo "[scale] model=$M seed=$si done=$(date)"
