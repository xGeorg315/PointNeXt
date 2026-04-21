#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/beegfs/scratch/workspace/es_gemiit01-pointnext_a42/PointNeXt}"
DATA_ROOT="${DATA_ROOT:-/beegfs/scratch/workspace/es_gemiit01-pointnext_a42/handcrafted_dataset_v1}"
CFG_PATH="${CFG_PATH:-${PROJECT_ROOT}/cfgs/modelnet40ply2048/pointnext-s.yaml}"

FAST_RUN_EPOCHS="${FAST_RUN_EPOCHS:-20}"
FAST_RUN_TRAIN_BATCHES="${FAST_RUN_TRAIN_BATCHES:-4}"
FAST_RUN_VAL_BATCHES="${FAST_RUN_VAL_BATCHES:-2}"
FAST_RUN_TEST_BATCHES="${FAST_RUN_TEST_BATCHES:-2}"
RUN_NAME="${RUN_NAME:-}"

if [[ -z "${RUN_NAME}" ]]; then
  echo "ERROR: RUN_NAME is required. Start with examples/classification/submit_fast_run.sh or export RUN_NAME before sbatch." >&2
  exit 2
fi

python "${PROJECT_ROOT}/examples/classification/main.py" \
  --cfg "${CFG_PATH}" \
  custom_dataset_root="${DATA_ROOT}" \
  wandb.name="${RUN_NAME}" \
  fast_run=True \
  fast_run_epochs="${FAST_RUN_EPOCHS}" \
  fast_run_train_batches="${FAST_RUN_TRAIN_BATCHES}" \
  fast_run_val_batches="${FAST_RUN_VAL_BATCHES}" \
  fast_run_test_batches="${FAST_RUN_TEST_BATCHES}" \
  root_dir=log_fast \
  num_points=512 \
  batch_size=8 \
  val_batch_size=8 \
  dataloader.num_workers=2 \
  print_freq=1 \
  val_freq=1 \
  wandb_vis_max_samples=4 \
  wandb_vis_max_wrong_samples=4 \
  wandb_vis_val_freq=1
