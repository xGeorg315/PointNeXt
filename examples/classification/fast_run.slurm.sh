#!/usr/bin/env bash
#SBATCH --job-name=pointnext-fast
#SBATCH --output=slurm_logs/%x.%j.out
#SBATCH --error=slurm_logs/%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2

set -euo pipefail

PROJECT_ROOT="/beegfs/scratch/workspace/es_gemiit01-pointnext_a42/PointNeXt"
FAST_RUN_SCRIPT="${PROJECT_ROOT}/examples/classification/fast_run.sh"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
source .venv/bin/activate

bash "${FAST_RUN_SCRIPT}"
