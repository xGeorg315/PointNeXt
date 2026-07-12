#!/usr/bin/env bash
# Runs up to three trainings concurrently on one physical GPU.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-mvf_10class_ablation_shared_encoder_gt_pose}"
LOG_DIR="${LOG_DIR:-log/mvf_shared_encoder_gt_pose_gpu${GPU_ID}}"
FAILURES_FILE="${LOG_DIR}/failed_runs.txt"
mkdir -p "${LOG_DIR}"
: > "${FAILURES_FILE}"

run_one() {
  local cfg="$1"
  local name="${cfg##*/}"
  name="${name%.yaml}"
  local log_file="${LOG_DIR}/${name}.log"

  echo "[$(date '+%F %T')] GPU ${GPU_ID}: starting ${cfg}"
  if CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" examples/classification/main.py --cfg "${cfg}" \
      wandb.use_wandb=True \
      "wandb.project=${WANDB_PROJECT}" >"${log_file}" 2>&1; then
    echo "[$(date '+%F %T')] GPU ${GPU_ID}: finished ${cfg}"
  else
    local status=$?
    echo "[$(date '+%F %T')] GPU ${GPU_ID}: FAILED (exit ${status}) ${cfg}" | tee -a "${FAILURES_FILE}"
  fi
}

worker() {
  local cfg
  for cfg in "$@"; do
    run_one "${cfg}"
  done
}

# Eight runs, distributed across three independent serial workers.
worker \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/01_baseline_5views_512_geometry_gradients.yaml" \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/04_5views_256_geometry_gradients.yaml" \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/08_5views_small_encoder.yaml" &
pids[0]=$!

worker \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/02_2views_512_geometry_gradients.yaml" \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/05_5views_1024_geometry_gradients.yaml" \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/09_5views_large_encoder.yaml" &
pids[1]=$!

worker \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/03_3views_512_geometry_gradients.yaml" \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/07_5views_detach_geometry.yaml" &
pids[2]=$!

for pid in "${pids[@]}"; do
  wait "${pid}" || true
done

echo "[$(date '+%F %T')] GPU ${GPU_ID}: queue complete. Failures: ${FAILURES_FILE}"
