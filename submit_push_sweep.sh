#!/bin/bash
#SBATCH --account=def-cneary
#SBATCH --job-name=push_confirm
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --array=0-20
#SBATCH --output=logs/%x-%A_%a.out

module load mujoco
source ~/project/.venv/bin/activate

mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python run_push_sweep.py \
  --batch /path/to/push_batch.jsonl \
  --index ${SLURM_ARRAY_TASK_ID} \
  --outdir /path/to/results_push_sweep \
  --steps 300000

