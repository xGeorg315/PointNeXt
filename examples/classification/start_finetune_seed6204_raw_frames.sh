#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

RUN_ID="modelnet40ply2048-train-pointnext-normal-ngpus1-seed6204-20260519-230551-Y37L8SA96BNSSrYywwVe5Z"
DEFAULT_CKPT="${PROJECT_ROOT}/log/modelnet40ply2048/${RUN_ID}/checkpoint/${RUN_ID}_ckpt_best.pth"

CFG_PATH="${CFG_PATH:-${PROJECT_ROOT}/cfgs/modelnet40ply2048/pointnext-raw-frames.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/georg/review-dataset-v3-split/raw_frames}"
CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"

SEED="${SEED:-6204}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-0.0001}"
FINETUNE_MODE="${FINETUNE_MODE:-finetune}"
FREEZE_ENCODER="${FREEZE_ENCODER:-False}"
BATCH_SIZE="${BATCH_SIZE:-18}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-18}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WANDB_USE_WANDB="${WANDB_USE_WANDB:-True}"
RAW_FRAMES_NUM_CLASSES="${RAW_FRAMES_NUM_CLASSES:-6}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

extra_args=()
if [[ -n "${EXTRA_ARGS}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_PATH}" >&2
  exit 2
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: raw-frames dataset root not found: ${DATA_ROOT}" >&2
  exit 2
fi

if (( BATCH_SIZE < RAW_FRAMES_NUM_CLASSES || BATCH_SIZE % RAW_FRAMES_NUM_CLASSES != 0 )); then
  echo "ERROR: BATCH_SIZE must be a multiple of RAW_FRAMES_NUM_CLASSES." >&2
  echo "       got BATCH_SIZE=${BATCH_SIZE}, RAW_FRAMES_NUM_CLASSES=${RAW_FRAMES_NUM_CLASSES}" >&2
  exit 2
fi

if (( VAL_BATCH_SIZE < RAW_FRAMES_NUM_CLASSES || VAL_BATCH_SIZE % RAW_FRAMES_NUM_CLASSES != 0 )); then
  echo "ERROR: VAL_BATCH_SIZE must be a multiple of RAW_FRAMES_NUM_CLASSES." >&2
  echo "       got VAL_BATCH_SIZE=${VAL_BATCH_SIZE}, RAW_FRAMES_NUM_CLASSES=${RAW_FRAMES_NUM_CLASSES}" >&2
  exit 2
fi

echo "Finetuning seed6204 PointNeXt on native 6-class raw_frames"
echo "  checkpoint: ${CKPT_PATH}"
echo "  data root:  ${DATA_ROOT}"
echo "  cfg:        ${CFG_PATH}"
echo "  seed:       ${SEED}"
echo "  epochs:     ${EPOCHS}"
echo "  lr:         ${LR}"
echo "  mode:       ${FINETUNE_MODE}"
echo "  freeze enc: ${FREEZE_ENCODER}"

python "${PROJECT_ROOT}/examples/classification/main.py" \
  --cfg "${CFG_PATH}" \
  mode="${FINETUNE_MODE}" \
  pretrained_path="${CKPT_PATH}" \
  finetune_freeze_encoder="${FREEZE_ENCODER}" \
  classification_dataset_format=raw_frames \
  raw_frames_root="${DATA_ROOT}" \
  custom_dataset_root="${DATA_ROOT}" \
  raw_frames_min_points=0 \
  raw_frames_split_ratios="[0.8,0.1,0.1]" \
  raw_frames_exclude_classes="[reject,TLS_VEHICLE_CAR_WITH_TRAILER,TLS_VEHICLE_TRUCK_WITH_TRAILER]" \
  class_balanced_batches=True \
  batch_size="${BATCH_SIZE}" \
  val_batch_size="${VAL_BATCH_SIZE}" \
  dataloader.num_workers="${NUM_WORKERS}" \
  epochs="${EPOCHS}" \
  lr="${LR}" \
  val_freq=1 \
  wandb.use_wandb="${WANDB_USE_WANDB}" \
  wandb_log_confusion_matrices=False \
  wandb_log_pcd_examples=False \
  wandb_vis_max_samples=0 \
  wandb_vis_max_wrong_samples=0 \
  seed="${SEED}" \
  "${extra_args[@]}" \
  "$@"
