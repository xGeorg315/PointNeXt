#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

read -r -p "Run name: " RUN_NAME
while [[ -z "${RUN_NAME//[[:space:]]/}" ]]; do
  echo "Run name must not be empty." >&2
  read -r -p "Run name: " RUN_NAME
done

SLURM_RUN_NAME="$(printf '%s' "${RUN_NAME}" | tr -cs '[:alnum:]_.-' '-')"
SLURM_RUN_NAME="${SLURM_RUN_NAME#-}"
SLURM_RUN_NAME="${SLURM_RUN_NAME%-}"
SLURM_RUN_NAME="${SLURM_RUN_NAME:-run}"

export RUN_NAME
cd "${PROJECT_ROOT}"
sbatch --job-name="pointnext-train-${SLURM_RUN_NAME:0:32}" "${SCRIPT_DIR}/start_train.slurm.sh"
