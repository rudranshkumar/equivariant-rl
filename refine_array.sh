#!/bin/bash
#SBATCH --account=def-cneary
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --array=0-89
#SBATCH --output=/home/klukasd/project/equivariant-rl/logs/%x-%A_%a.out

# ---------- environment ----------
module load mujoco
source /home/klukasd/project/.venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ---------- paths ----------
PROJECT_DIR=/home/klukasd/project/equivariant-rl
BATCH_FILE=${PROJECT_DIR}/refine_batch.jsonl
OUT_DIR=${PROJECT_DIR}/results

mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${OUT_DIR}

# ---------- run one trial ----------
python ${PROJECT_DIR}/run_one_trial.py \
  --batch ${BATCH_FILE} \
  --index ${SLURM_ARRAY_TASK_ID} \
  --outdir ${OUT_DIR} \
  --steps 300000
