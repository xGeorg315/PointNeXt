#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/home/georg/review-dataset}"
CFG_PATH="${CFG_PATH:-${PROJECT_ROOT}/cfgs/modelnet40ply2048/pointnext-completion-cls.yaml}"

RUN_NAME="${RUN_NAME:-review_symmetry_completion_$(date +%Y%m%d_%H%M%S)}"
REVIEW_START_DATE="${REVIEW_START_DATE:-2026-05-05}"
REVIEW_MIN_POINTS="${REVIEW_MIN_POINTS:-1024}"
REVIEW_SPLIT_RATIOS="${REVIEW_SPLIT_RATIOS:-[0.8,0.1,0.1]}"
REVIEW_EXCLUDE_CLASSES="${REVIEW_EXCLUDE_CLASSES:-[reject]}"

NUM_POINTS="${NUM_POINTS:-512}"
NUM_COMPLETE="${NUM_COMPLETE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-27}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-9}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-100}"

SYMMETRY_AXIS="${SYMMETRY_AXIS:-x}"
SYMMETRY_SOURCE="${SYMMETRY_SOURCE:-partial}"
SYMMETRY_KEEP_SIDE="${SYMMETRY_KEEP_SIDE:-auto}"

WANDB_USE_WANDB="${WANDB_USE_WANDB:-True}"
WANDB_VIS_MAX_SAMPLES="${WANDB_VIS_MAX_SAMPLES:-4}"
SAVE_WANDB_POINTCLOUDS_LOCAL="${SAVE_WANDB_POINTCLOUDS_LOCAL:-True}"
LOCAL_POINTCLOUD_DIR="${LOCAL_POINTCLOUD_DIR:-}"

python "${PROJECT_ROOT}/examples/classification/main.py" \
  --cfg "${CFG_PATH}" \
  task=completion_cls \
  custom_dataset_root="${DATA_ROOT}" \
  completion_dataset_format=review \
  review_start_date="${REVIEW_START_DATE}" \
  review_min_points="${REVIEW_MIN_POINTS}" \
  review_split_ratios="${REVIEW_SPLIT_RATIOS}" \
  review_exclude_classes="${REVIEW_EXCLUDE_CLASSES}" \
  completion_symmetry=True \
  completion_symmetry_axis="${SYMMETRY_AXIS}" \
  completion_symmetry_source="${SYMMETRY_SOURCE}" \
  completion_symmetry_keep_side="${SYMMETRY_KEEP_SIDE}" \
  num_points="${NUM_POINTS}" \
  num_complete="${NUM_COMPLETE}" \
  model.decoder_args.num_fine="${NUM_COMPLETE}" \
  batch_size="${BATCH_SIZE}" \
  val_batch_size="${VAL_BATCH_SIZE}" \
  dataloader.num_workers="${NUM_WORKERS}" \
  epochs="${EPOCHS}" \
  val_freq=1 \
  wandb.use_wandb="${WANDB_USE_WANDB}" \
  wandb.name="${RUN_NAME}" \
  wandb_vis_max_samples="${WANDB_VIS_MAX_SAMPLES}" \
  save_wandb_pointclouds_local="${SAVE_WANDB_POINTCLOUDS_LOCAL}" \
  local_pointcloud_dir="${LOCAL_POINTCLOUD_DIR}"
