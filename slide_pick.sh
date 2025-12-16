#!/bin/bash
#SBATCH --account=def-cneary
#SBATCH --job-name=ps_sweep
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --array=0-79
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --signal=B:USR1@60

module load mujoco
source ~/project/.venv/bin/activate

mkdir -p logs

# Keep threads from fighting you
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

BATCH=/home/klukasd/project/equivariant-rl/pick_slide_sweep.jsonl
OUTDIR=/home/klukasd/project/equivariant-rl/results_ps_sweep

python run_one_trial.py \
  --batch "$BATCH" \
  --index ${SLURM_ARRAY_TASK_ID} \
  --outdir "$OUTDIR" \
  --steps 1000000

