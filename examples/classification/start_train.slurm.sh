#!/usr/bin/env bash
#SBATCH --job-name=pointnext-train
#SBATCH --output=slurm_logs/%x.%j.out
#SBATCH --error=slurm_logs/%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6

set -euo pipefail

PROJECT_ROOT="/beegfs/scratch/workspace/es_gemiit01-pointnext_a42/PointNeXt"
TRAIN_SCRIPT="${PROJECT_ROOT}/examples/classification/start_train.sh"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs

# Optional: only enable if your cluster needs modules/conda activation.
# module load cuda/11.1.1
# module load gcc
source .venv/bin/activate

bash "${TRAIN_SCRIPT}"
