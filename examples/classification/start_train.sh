#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-}"

if [[ -z "${RUN_NAME}" ]]; then
  echo "ERROR: RUN_NAME is required. Start with examples/classification/submit_train.sh or export RUN_NAME before sbatch." >&2
  exit 2
fi

python /beegfs/scratch/workspace/es_gemiit01-pointnext_a42/PointNeXt/examples/classification/main.py \
  --cfg /beegfs/scratch/workspace/es_gemiit01-pointnext_a42/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml \
  custom_dataset_root=/beegfs/scratch/workspace/es_gemiit01-pointnext_a42/handcrafted_dataset_v1 \
  wandb.name="${RUN_NAME}" \
  dataloader.num_workers=6
