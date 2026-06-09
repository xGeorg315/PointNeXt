#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

RUN_ID="modelnet40ply2048-train-pointnext-normal-ngpus1-seed6204-20260519-230551-Y37L8SA96BNSSrYywwVe5Z"
DEFAULT_CKPT="${PROJECT_ROOT}/log/modelnet40ply2048/${RUN_ID}/checkpoint/${RUN_ID}_ckpt_best.pth"

TEST_DATASET="${TEST_DATASET:-pcd}"  # pcd or raw_frames
CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-18}"
WANDB_USE_WANDB="${WANDB_USE_WANDB:-False}"
SEED="${SEED:-847}"
RAW_FRAMES_CLASS_MODE="${RAW_FRAMES_CLASS_MODE:-auto}"  # auto, 6, or 7

COMMON_ARGS=(
  mode=test
  pretrained_path="${CKPT_PATH}"
  dataloader.num_workers="${NUM_WORKERS}"
  val_batch_size="${VAL_BATCH_SIZE}"
  class_balanced_batches=True
  wandb.use_wandb="${WANDB_USE_WANDB}"
  wandb_log_confusion_matrices=False
  wandb_log_pcd_examples=False
  wandb_vis_max_samples=0
  wandb_vis_max_wrong_samples=0
  seed="${SEED}"
)

case "${TEST_DATASET}" in
  pcd)
    CFG_PATH="${CFG_PATH:-${PROJECT_ROOT}/cfgs/modelnet40ply2048/pointnext-normal.yaml}"
    DATA_ROOT="${DATA_ROOT:-/home/georg/review-dataset-v3-split/pcd}"
    DATA_ARGS=(
      classification_dataset_format=review
      custom_dataset_root="${DATA_ROOT}"
      review_dataset_root="${DATA_ROOT}"
      review_min_points=0
      review_split_ratios="[0.8,0.1,0.1]"
      exclude_classes="[TLS_VEHICLE_CAR_WITH_TRAILER,TLS_VEHICLE_TRUCK_WITH_TRAILER]"
    )
    ;;
  raw_frames)
    CFG_PATH="${CFG_PATH:-${PROJECT_ROOT}/cfgs/modelnet40ply2048/pointnext-raw-frames.yaml}"
    DATA_ROOT="${DATA_ROOT:-/home/georg/review-dataset-v3-split/raw_frames}"
    DATA_ARGS=(
      classification_dataset_format=raw_frames
      raw_frames_root="${DATA_ROOT}"
      custom_dataset_root="${DATA_ROOT}"
      raw_frames_min_points=0
      raw_frames_split_ratios="[0.8,0.1,0.1]"
      raw_frames_exclude_classes="[reject,TLS_VEHICLE_CAR_WITH_TRAILER,TLS_VEHICLE_TRUCK_WITH_TRAILER]"
    )
    resolved_raw_frames_class_mode="${RAW_FRAMES_CLASS_MODE}"
    if [[ "${resolved_raw_frames_class_mode}" == "auto" ]]; then
      if [[ "${CKPT_PATH}" == *"seed847-20260522"* ]]; then
        resolved_raw_frames_class_mode="7"
      else
        resolved_raw_frames_class_mode="6"
      fi
    fi

    case "${resolved_raw_frames_class_mode}" in
      7)
        DATA_ARGS+=(
          raw_frames_classes="[TLS_VEHICLE_BUS,TLS_VEHICLE_CAR,TLS_VEHICLE_MOTORBIKE,TLS_VEHICLE_SEMI_TRAILER_TRUCK,TLS_VEHICLE_TRAILER,TLS_VEHICLE_TRUCK,TLS_VEHICLE_VAN]"
        )
        echo "Using 7-class raw_frames label map; TLS_VEHICLE_TRAILER may have no raw_frames samples." >&2
        ;;
      6)
        echo "Using native 6-class raw_frames label map." >&2
        ;;
      *)
        echo "ERROR: RAW_FRAMES_CLASS_MODE must be auto, 6, or 7 (got ${RAW_FRAMES_CLASS_MODE})." >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "ERROR: TEST_DATASET must be 'pcd' or 'raw_frames' (got '${TEST_DATASET}')." >&2
    exit 2
    ;;
esac

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_PATH}" >&2
  exit 2
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: dataset root not found: ${DATA_ROOT}" >&2
  exit 2
fi

echo "Testing ${RUN_ID}"
echo "  checkpoint: ${CKPT_PATH}"
echo "  dataset:    ${TEST_DATASET}"
echo "  data root:  ${DATA_ROOT}"
echo "  cfg:        ${CFG_PATH}"

python "${PROJECT_ROOT}/examples/classification/main.py" \
  --cfg "${CFG_PATH}" \
  "${COMMON_ARGS[@]}" \
  "${DATA_ARGS[@]}"
